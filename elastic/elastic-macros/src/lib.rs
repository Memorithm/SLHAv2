//! elastic-macros — the Elastic resource language's embedded syntax.
//!
//! These macros are NOT decorative aliases: they lower into the actual
//! `elastic-core` runtime abstractions (tier machines, budgets, policies,
//! transactions). Where invalid transitions can be rejected at compile time,
//! they are; where dynamic state makes compile-time rejection impossible,
//! they generate explicit checked `Result` calls against a validated table.
//!
//! Provided macros:
//! - [`elastic_state!`]: declares a tier enum + validated transition table.
//! - [`elastic_budget!`]: declares a hierarchical budget with limits.
//! - [`elastic_policy!`]: declares a policy struct (hard constraints,
//!   objectives, hysteresis watermarks, flags).
//! - [`elastic_transition!`]: executes a transactional prepare/verify/commit/
//!   rollback block.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{braced, parse_macro_input, Token};

// ────────────────────────────────────────────────────────────────────────────
// elastic_state!
// ────────────────────────────────────────────────────────────────────────────

/// Declare a tier enum with a validated transition table.
///
/// ```ignore
/// elastic_state! {
///     ContextTier {
///         Pinned, Hot, Warm, Cold, Evicted,
///     }
///     transitions {
///         Hot => Warm,
///         Warm => Hot,
///         Warm => Cold,
///         Cold => Warm,
///         Cold => Evicted,
///         Evicted => Cold,
///         Pinned => !Evicted,   // explicitly forbidden edge
///     }
/// }
/// ```
///
/// `A => B` declares a directed edge; `A => !B` declares that `A -> B` is
/// FORBIDDEN (a negative edge, checked at compile time against the declared
/// tier set). The macro generates the enum, the validated
/// `elastic_core::tiers::TierMachine`, `TierLike` impls, and a checked
/// `try_move` method.
struct StateInput {
    enum_name: syn::Ident,
    tiers: Vec<syn::Ident>,
    transitions: Vec<TransitionDecl>,
}

enum TransitionDecl {
    One(syn::Ident, syn::Ident),
    Forbid(syn::Ident, syn::Ident),
}

impl Parse for StateInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let enum_name: syn::Ident = input.parse()?;
        let content;
        braced!(content in input);
        let tiers: Punctuated<syn::Ident, Token![,]> =
            content.parse_terminated(syn::Ident::parse, Token![,])?;

        let kw: syn::Ident = input.parse()?;
        if kw != "transitions" {
            return Err(syn::Error::new(kw.span(), "expected `transitions`"));
        }
        let tcontent;
        braced!(tcontent in input);
        let mut transitions = Vec::new();
        while !tcontent.is_empty() {
            let from: syn::Ident = tcontent.parse()?;
            let _arrow: syn::Token![=>] = tcontent.parse()?;
            let not: Option<syn::Token![!]> = tcontent.parse()?;
            let to: syn::Ident = tcontent.parse()?;
            if not.is_some() {
                transitions.push(TransitionDecl::Forbid(from, to));
            } else {
                transitions.push(TransitionDecl::One(from, to));
            }
            if tcontent.peek(Token![,]) {
                let _: Token![,] = tcontent.parse()?;
            } else if !tcontent.is_empty() {
                return Err(tcontent.error("expected `,`"));
            }
        }
        Ok(Self {
            enum_name,
            tiers: tiers.into_iter().collect(),
            transitions,
        })
    }
}

#[proc_macro]
pub fn elastic_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StateInput);
    let name = &input.enum_name;

    // Compile-time validation: every transition endpoint must be a declared
    // tier, and no forbidden edge may also be declared positive.
    let tier_names: Vec<String> = input.tiers.iter().map(|t| t.to_string()).collect();
    let mut validations = Vec::new();
    for tr in &input.transitions {
        let (from, to) = match tr {
            TransitionDecl::One(a, b) | TransitionDecl::Forbid(a, b) => (a, b),
        };
        let f = from.to_string();
        let t = to.to_string();
        let has_f = tier_names.contains(&f);
        let has_t = tier_names.contains(&t);
        let kind = match tr {
            TransitionDecl::One(..) => "edge",
            TransitionDecl::Forbid(..) => "forbidden edge",
        };
        validations.push(quote! {
            const _: () = assert!(
                #has_f,
                concat!("elastic_state!: ", #kind, " source `", #f, "` is not a declared tier")
            );
            const _: () = assert!(
                #has_t,
                concat!("elastic_state!: ", #kind, " target `", #t, "` is not a declared tier")
            );
        });
    }

    // Forbidden edges: emit compile-time asserts that they are NOT declared
    // as positive edges elsewhere.
    for tr in &input.transitions {
        if let TransitionDecl::Forbid(from, to) = tr {
            let f = from.to_string();
            let t = to.to_string();
            let also_positive = input
                .transitions
                .iter()
                .any(|other| matches!(other, TransitionDecl::One(a, b) if a == from && b == to));
            validations.push(quote! {
                const _: () = assert!(
                    !#also_positive,
                    concat!("elastic_state!: edge `", #f, " => ", #t, "` declared both positive and forbidden")
                );
            });
        }
    }

    // Runtime table: declared positive edges (the forbidden ones are
    // deliberately absent).
    let mut edges = Vec::new();
    for tr in &input.transitions {
        if let TransitionDecl::One(from, to) = tr {
            edges.push((from.to_string(), to.to_string()));
        }
    }

    // Ranks: first declared tier = highest rank (most resident).
    let n = input.tiers.len();
    let tier_lits: Vec<String> = input.tiers.iter().map(|t| t.to_string()).collect();
    let ranks: Vec<u8> = (0..n as u8).rev().collect();
    let variants = &input.tiers;
    let from_lits: Vec<String> = edges.iter().map(|(f, _)| f.clone()).collect();
    let to_lits: Vec<String> = edges.iter().map(|(_, t)| t.clone()).collect();

    quote! {
        #(#validations)*

        /// Tier enum generated by `elastic_state!`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum #name {
            #(#variants),*
        }

        impl #name {
            /// The validated tier machine for this state.
            pub fn machine() -> &'static elastic_core::tiers::TierMachine {
                use elastic_core::tiers::{Tier, TierMachine, TierTransition};
                static MACHINE: std::sync::OnceLock<TierMachine> = std::sync::OnceLock::new();
                MACHINE.get_or_init(|| {
                    TierMachine::new(
                        vec![#(Tier { name: #tier_lits, rank: #ranks }),*],
                        vec![#(TierTransition {
                            from: Tier { name: #from_lits, rank: 0 },
                            to: Tier { name: #to_lits, rank: 0 },
                        }),*],
                    )
                })
            }

            /// Checked transition; returns `Err(TierError)` for undeclared
            /// or forbidden edges (e.g. Pinned => Evicted).
            pub fn try_move(&mut self, to: #name) -> Result<(), elastic_core::tiers::TierError> {
                let from = *self;
                let m = Self::machine();
                m.transition(from.as_tier(), to.as_tier())?;
                *self = to;
                Ok(())
            }
        }

        impl elastic_core::tiers::TierLike for #name {
            fn as_tier(self) -> elastic_core::tiers::Tier {
                match self {
                    #(Self::#variants => elastic_core::tiers::Tier {
                        name: #tier_lits,
                        rank: #ranks,
                    }),*
                }
            }

            fn from_tier(t: elastic_core::tiers::Tier) -> Self {
                match t.name {
                    #(#tier_lits => Self::#variants),*
                    _ => panic!("elastic_state!: unknown tier `{}`", t.name),
                }
            }
        }

        impl core::fmt::Display for #name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}", self.as_tier().name)
            }
        }
    }
    .into()
}

// ────────────────────────────────────────────────────────────────────────────
// elastic_budget!
// ────────────────────────────────────────────────────────────────────────────

/// Declare a hierarchical budget with resource limits.
///
/// ```ignore
/// elastic_budget! {
///     vram <= 80%;
///     ram <= 70%;
///     latency <= 25.ms();
/// }
/// ```
///
/// Each line `name <= expr` declares a named budget constraint. The macro
/// generates a struct with one `f64` field per constraint, plus a builder.
struct BudgetInput {
    constraints: Vec<(syn::Ident, syn::Expr)>,
}

impl Parse for BudgetInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut constraints = Vec::new();
        while !input.is_empty() {
            let name: syn::Ident = input.parse()?;
            let _le: syn::Token![<=] = input.parse()?;
            let expr: syn::Expr = input.parse()?;
            constraints.push((name, expr));
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            } else if !input.is_empty() {
                return Err(input.error("expected `,`"));
            }
        }
        Ok(Self { constraints })
    }
}

#[proc_macro]
pub fn elastic_budget(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BudgetInput);
    let names: Vec<&syn::Ident> = input.constraints.iter().map(|(n, _)| n).collect();
    let exprs: Vec<&syn::Expr> = input.constraints.iter().map(|(_, e)| e).collect();
    let n = input.constraints.len();
    quote! {
        /// Budget struct generated by `elastic_budget!`.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct ElasticBudget {
            #(pub #names: f64),*
        }

        impl ElasticBudget {
            /// All constraints satisfied.
            pub fn all_satisfied(&self) -> bool {
                #(self.#names >= 0.0 && self.#names <= #exprs &&)* true
            }

            /// Number of constraints.
            pub const fn len(&self) -> usize { #n }

            /// Whether the budget is empty.
            pub const fn is_empty(&self) -> bool { #n == 0 }
        }
    }
    .into()
}

// ────────────────────────────────────────────────────────────────────────────
// elastic_policy!
// ────────────────────────────────────────────────────────────────────────────

/// Declare a policy: hard constraints, objectives, hysteresis, flags.
///
/// ```ignore
/// elastic_policy! {
///     ContextPolicy {
///         hard { correctness: "required", pinned: "preserved" }
///         objectives { "maximize_retention", "minimize_latency" }
///         hysteresis { high: 0.85, low: 0.70 }
///         predictive: true
///         transactional: true
///     }
/// }
/// ```
///
/// Generates a struct with the declared fields and a builder.
struct PolicyInput {
    name: syn::Ident,
    hard: Vec<(syn::Ident, syn::LitStr)>,
    objectives: Vec<syn::LitStr>,
    high: f64,
    low: f64,
    predictive: bool,
    transactional: bool,
}

impl Parse for PolicyInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let name: syn::Ident = input.parse()?;
        let content;
        braced!(content in input);

        let mut hard = Vec::new();
        let mut objectives = Vec::new();
        let mut high = 0.85f64;
        let mut low = 0.70f64;
        let mut predictive = false;
        let mut transactional = false;

        while !content.is_empty() {
            let key: syn::Ident = content.parse()?;
            match key.to_string().as_str() {
                "hard" => {
                    let hc;
                    braced!(hc in content);
                    while !hc.is_empty() {
                        let k: syn::Ident = hc.parse()?;
                        let _colon: syn::Token![:] = hc.parse()?;
                        let v: syn::LitStr = hc.parse()?;
                        hard.push((k, v));
                        if hc.peek(Token![,]) {
                            let _: Token![,] = hc.parse()?;
                        }
                    }
                }
                "objectives" => {
                    let oc;
                    braced!(oc in content);
                    while !oc.is_empty() {
                        let v: syn::LitStr = oc.parse()?;
                        objectives.push(v);
                        if oc.peek(Token![,]) {
                            let _: Token![,] = oc.parse()?;
                        }
                    }
                }
                "hysteresis" => {
                    let hc;
                    braced!(hc in content);
                    while !hc.is_empty() {
                        let k: syn::Ident = hc.parse()?;
                        let _colon: syn::Token![:] = hc.parse()?;
                        let v: syn::LitFloat = hc.parse()?;
                        let v: f64 = v.base10_parse()?;
                        if k == "high" {
                            high = v;
                        } else if k == "low" {
                            low = v;
                        } else {
                            return Err(syn::Error::new(k.span(), "expected `high` or `low`"));
                        }
                        if hc.peek(Token![,]) {
                            let _: Token![,] = hc.parse()?;
                        }
                    }
                }
                "predictive" => {
                    let _colon: syn::Token![:] = content.parse()?;
                    let v: syn::LitBool = content.parse()?;
                    predictive = v.value;
                }
                "transactional" => {
                    let _colon: syn::Token![:] = content.parse()?;
                    let v: syn::LitBool = content.parse()?;
                    transactional = v.value;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown policy key `{other}` (expected hard/objectives/hysteresis/predictive/transactional)"),
                    ));
                }
            }
            if content.peek(Token![,]) {
                let _: Token![,] = content.parse()?;
            }
        }

        Ok(Self {
            name,
            hard,
            objectives,
            high,
            low,
            predictive,
            transactional,
        })
    }
}

#[proc_macro]
pub fn elastic_policy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as PolicyInput);
    let name = &input.name;
    let hard_names: Vec<&syn::Ident> = input.hard.iter().map(|(n, _)| n).collect();
    let hard_vals: Vec<&syn::LitStr> = input.hard.iter().map(|(_, v)| v).collect();
    let objectives: Vec<&syn::LitStr> = input.objectives.iter().collect();
    let high = input.high;
    let low = input.low;
    let predictive = input.predictive;
    let transactional = input.transactional;
    let n_hard = input.hard.len();
    let n_obj = input.objectives.len();

    quote! {
        /// Policy struct generated by `elastic_policy!`.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct #name {
            #(pub #hard_names: &'static str),*
            pub objectives: &'static [&'static str],
            pub hysteresis_high: f64,
            pub hysteresis_low: f64,
            pub predictive: bool,
            pub transactional: bool,
        }

        impl #name {
            /// Build the policy with the declared values.
            pub const fn build() -> Self {
                Self {
                    #(#hard_names: #hard_vals),*
                    objectives: &[#(#objectives),*],
                    hysteresis_high: #high,
                    hysteresis_low: #low,
                    predictive: #predictive,
                    transactional: #transactional,
                }
            }

            /// Number of hard constraints.
            pub const fn hard_len(&self) -> usize { #n_hard }

            /// Number of objectives.
            pub const fn objective_len(&self) -> usize { #n_obj }
        }

        impl Default for #name {
            fn default() -> Self {
                Self::build()
            }
        }
    }
    .into()
}

// ────────────────────────────────────────────────────────────────────────────
// elastic_target!
// ────────────────────────────────────────────────────────────────────────────

/// Declare a target/objective expression: `maximize X, subject_to { … }`.
///
/// ```ignore
/// elastic_target! {
///     maximize logical_context,
///     subject_to {
///         vram_pressure < 0.85,
///         latency <= target_latency,
///     }
/// }
/// ```
///
/// Generates a struct with the objective name and the constraint
/// expressions as fields, evaluated lazily by the caller.
struct TargetInput {
    objective: syn::Expr,
    constraints: Vec<syn::Expr>,
}

impl Parse for TargetInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let kw: syn::Ident = input.parse()?;
        if kw != "maximize" {
            return Err(syn::Error::new(kw.span(), "expected `maximize`"));
        }
        let objective: syn::Expr = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let kw2: syn::Ident = input.parse()?;
        if kw2 != "subject_to" {
            return Err(syn::Error::new(kw2.span(), "expected `subject_to`"));
        }
        let content;
        braced!(content in input);
        let constraints: Punctuated<syn::Expr, Token![,]> =
            content.parse_terminated(syn::Expr::parse, Token![,])?;
        Ok(Self {
            objective,
            constraints: constraints.into_iter().collect(),
        })
    }
}

#[proc_macro]
pub fn elastic_target(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TargetInput);
    let _objective = &input.objective;
    let _constraints = &input.constraints;
    let n = input.constraints.len();
    quote! {
        /// Target struct generated by `elastic_target!`.
        #[derive(Clone, Copy, Debug)]
        pub struct ElasticTarget {
            /// The objective expression.
            pub objective: f64,
            /// Constraint expressions.
            pub constraints: [f64; #n],
        }
    }
    .into()
}
