//! Procedural macros for the Elastic resource language.
//!
//! The macros deliberately separate executable runtime state from descriptive
//! policy metadata:
//! - `elastic_state!` generates a checked tier machine;
//! - `elastic_budget!` generates an executable set of numeric limits;
//! - `elastic_policy!` generates validated policy metadata (hard-constraint
//!   labels/objective labels plus controller settings); actual hard-constraint
//!   enforcement remains the responsibility of `ElasticConstraints`/the
//!   resource backend;
//! - `elastic_target!` evaluates a numeric objective and boolean constraints at
//!   the call site and returns a target value whose feasibility is inspectable.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{braced, parse_macro_input, Token};

// ---------------------------------------------------------------------------
// elastic_state!
// ---------------------------------------------------------------------------

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

        let keyword: syn::Ident = input.parse()?;
        if keyword != "transitions" {
            return Err(syn::Error::new(keyword.span(), "expected `transitions`"));
        }
        let transition_content;
        braced!(transition_content in input);
        let mut transitions = Vec::new();
        while !transition_content.is_empty() {
            let from: syn::Ident = transition_content.parse()?;
            let _: syn::Token![=>] = transition_content.parse()?;
            let forbidden: Option<syn::Token![!]> = transition_content.parse()?;
            let to: syn::Ident = transition_content.parse()?;
            transitions.push(if forbidden.is_some() {
                TransitionDecl::Forbid(from, to)
            } else {
                TransitionDecl::One(from, to)
            });
            if transition_content.peek(Token![,]) {
                let _: Token![,] = transition_content.parse()?;
            } else if !transition_content.is_empty() {
                return Err(transition_content.error("expected `,`"));
            }
        }

        Ok(Self {
            enum_name,
            tiers: tiers.into_iter().collect(),
            transitions,
        })
    }
}

/// Declare a tier enum and its directed transition table.
///
/// `A => B` is an allowed transition; `A => !B` is an explicitly forbidden
/// edge. Endpoints, duplicate tiers, graph conflicts and the `u8` rank limit
/// are checked while the procedural macro expands.
#[proc_macro]
pub fn elastic_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as StateInput);
    let name = &input.enum_name;

    if input.tiers.is_empty() {
        return syn::Error::new(name.span(), "elastic_state!: at least one tier is required")
            .to_compile_error()
            .into();
    }
    if input.tiers.len() > 256 {
        return syn::Error::new(
            name.span(),
            "elastic_state!: at most 256 tiers are supported by the u8 rank representation",
        )
        .to_compile_error()
        .into();
    }

    let tier_names: Vec<String> = input.tiers.iter().map(ToString::to_string).collect();
    for (index, tier) in tier_names.iter().enumerate() {
        if tier_names[..index].contains(tier) {
            return syn::Error::new(
                input.tiers[index].span(),
                format!("elastic_state!: duplicate tier `{tier}`"),
            )
            .to_compile_error()
            .into();
        }
    }

    let mut positive_edges: Vec<(String, String)> = Vec::new();
    let mut forbidden_edges: Vec<(String, String)> = Vec::new();
    for transition in &input.transitions {
        let (from, to, forbidden) = match transition {
            TransitionDecl::One(from, to) => (from, to, false),
            TransitionDecl::Forbid(from, to) => (from, to, true),
        };
        let from_name = from.to_string();
        let to_name = to.to_string();
        if !tier_names.contains(&from_name) {
            return syn::Error::new(
                from.span(),
                format!("elastic_state!: source tier `{from_name}` is not declared"),
            )
            .to_compile_error()
            .into();
        }
        if !tier_names.contains(&to_name) {
            return syn::Error::new(
                to.span(),
                format!("elastic_state!: target tier `{to_name}` is not declared"),
            )
            .to_compile_error()
            .into();
        }
        let edge = (from_name, to_name);
        let destination = if forbidden {
            &mut forbidden_edges
        } else {
            &mut positive_edges
        };
        if destination.contains(&edge) {
            return syn::Error::new(
                from.span(),
                "elastic_state!: duplicate transition declaration",
            )
            .to_compile_error()
            .into();
        }
        destination.push(edge);
    }

    for edge in &positive_edges {
        if forbidden_edges.contains(edge) {
            return syn::Error::new(
                name.span(),
                format!(
                    "elastic_state!: edge `{} => {}` is both allowed and forbidden",
                    edge.0, edge.1
                ),
            )
            .to_compile_error()
            .into();
        }
    }

    let n = input.tiers.len();
    let variants = &input.tiers;
    let tier_literals = tier_names;
    let ranks: Vec<u8> = (0..n)
        .rev()
        .map(|rank| u8::try_from(rank).expect("tier count checked above"))
        .collect();
    let from_literals: Vec<String> = positive_edges
        .iter()
        .map(|(from, _)| from.clone())
        .collect();
    let to_literals: Vec<String> = positive_edges
        .iter()
        .map(|(_, to)| to.clone())
        .collect();
    let edge_count = positive_edges.len();

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum #name {
            #(#variants),*
        }

        impl #name {
            pub fn machine() -> &'static elastic_core::tiers::TierMachine {
                use elastic_core::tiers::{Tier, TierMachine, TierTransition};
                static MACHINE: std::sync::OnceLock<TierMachine> = std::sync::OnceLock::new();
                MACHINE.get_or_init(|| {
                    let mut tiers = Vec::with_capacity(#n);
                    #(
                        tiers.push(Tier { name: #tier_literals, rank: #ranks });
                    )*
                    let mut transitions = Vec::with_capacity(#edge_count);
                    #(
                        let from = tiers
                            .iter()
                            .copied()
                            .find(|tier| tier.name == #from_literals)
                            .expect("macro-validated source tier");
                        let to = tiers
                            .iter()
                            .copied()
                            .find(|tier| tier.name == #to_literals)
                            .expect("macro-validated target tier");
                        transitions.push(TierTransition { from, to });
                    )*
                    TierMachine::new(tiers, transitions)
                })
            }

            pub fn try_move(
                &mut self,
                to: #name,
            ) -> Result<(), elastic_core::tiers::TierError> {
                let from = *self;
                Self::machine().transition(
                    elastic_core::tiers::TierLike::as_tier(from),
                    elastic_core::tiers::TierLike::as_tier(to),
                )?;
                *self = to;
                Ok(())
            }
        }

        impl elastic_core::tiers::TierLike for #name {
            fn as_tier(self) -> elastic_core::tiers::Tier {
                match self {
                    #(
                        Self::#variants => elastic_core::tiers::Tier {
                            name: #tier_literals,
                            rank: #ranks,
                        },
                    )*
                }
            }

            fn from_tier(tier: elastic_core::tiers::Tier) -> Self {
                match tier.name {
                    #(#tier_literals => Self::#variants,)*
                    _ => panic!("elastic_state!: unknown tier `{}`", tier.name),
                }
            }
        }

        impl core::fmt::Display for #name {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let tier = elastic_core::tiers::TierLike::as_tier(*self);
                formatter.write_str(tier.name)
            }
        }
    }
    .into()
}

// ---------------------------------------------------------------------------
// elastic_budget!
// ---------------------------------------------------------------------------

struct BudgetInput {
    constraints: Vec<(syn::Ident, syn::Expr)>,
}

impl Parse for BudgetInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut constraints = Vec::new();
        while !input.is_empty() {
            let name: syn::Ident = input.parse()?;
            let _: syn::Token![<=] = input.parse()?;
            let expression: syn::Expr = input.parse()?;
            constraints.push((name, expression));
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            } else if !input.is_empty() {
                return Err(input.error("expected `,`"));
            }
        }
        Ok(Self { constraints })
    }
}

/// Declare named numeric upper bounds.
///
/// The generated `ElasticBudget` stores observed values; `all_satisfied()`
/// evaluates every declared upper-bound expression at the call site.
#[proc_macro]
pub fn elastic_budget(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BudgetInput);
    let names: Vec<&syn::Ident> = input.constraints.iter().map(|(name, _)| name).collect();
    let expressions: Vec<&syn::Expr> = input
        .constraints
        .iter()
        .map(|(_, expression)| expression)
        .collect();
    let count = input.constraints.len();

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct ElasticBudget {
            #(pub #names: f64),*
        }

        impl ElasticBudget {
            pub fn all_satisfied(&self) -> bool {
                #(self.#names.is_finite()
                    && self.#names >= 0.0
                    && self.#names <= (#expressions as f64)
                    &&)* true
            }

            pub const fn len(&self) -> usize {
                #count
            }

            pub const fn is_empty(&self) -> bool {
                #count == 0
            }
        }
    }
    .into()
}

// ---------------------------------------------------------------------------
// elastic_policy!
// ---------------------------------------------------------------------------

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
                    let hard_content;
                    braced!(hard_content in content);
                    while !hard_content.is_empty() {
                        let hard_name: syn::Ident = hard_content.parse()?;
                        let _: syn::Token![:] = hard_content.parse()?;
                        let value: syn::LitStr = hard_content.parse()?;
                        hard.push((hard_name, value));
                        if hard_content.peek(Token![,]) {
                            let _: Token![,] = hard_content.parse()?;
                        } else if !hard_content.is_empty() {
                            return Err(hard_content.error("expected `,`"));
                        }
                    }
                }
                "objectives" => {
                    let objective_content;
                    braced!(objective_content in content);
                    while !objective_content.is_empty() {
                        objectives.push(objective_content.parse()?);
                        if objective_content.peek(Token![,]) {
                            let _: Token![,] = objective_content.parse()?;
                        } else if !objective_content.is_empty() {
                            return Err(objective_content.error("expected `,`"));
                        }
                    }
                }
                "hysteresis" => {
                    let hysteresis_content;
                    braced!(hysteresis_content in content);
                    while !hysteresis_content.is_empty() {
                        let watermark: syn::Ident = hysteresis_content.parse()?;
                        let _: syn::Token![:] = hysteresis_content.parse()?;
                        let literal: syn::LitFloat = hysteresis_content.parse()?;
                        let value: f64 = literal.base10_parse()?;
                        match watermark.to_string().as_str() {
                            "high" => high = value,
                            "low" => low = value,
                            _ => {
                                return Err(syn::Error::new(
                                    watermark.span(),
                                    "expected `high` or `low`",
                                ));
                            }
                        }
                        if hysteresis_content.peek(Token![,]) {
                            let _: Token![,] = hysteresis_content.parse()?;
                        } else if !hysteresis_content.is_empty() {
                            return Err(hysteresis_content.error("expected `,`"));
                        }
                    }
                }
                "predictive" => {
                    let _: syn::Token![:] = content.parse()?;
                    predictive = content.parse::<syn::LitBool>()?.value;
                }
                "transactional" => {
                    let _: syn::Token![:] = content.parse()?;
                    transactional = content.parse::<syn::LitBool>()?.value;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown policy key `{other}` (expected hard/objectives/hysteresis/predictive/transactional)"
                        ),
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

/// Declare validated policy metadata.
///
/// The strings name hard constraints and objectives for telemetry/configuration;
/// they do not replace executable `ElasticConstraints` checks. Hysteresis is
/// validated during macro expansion.
#[proc_macro]
pub fn elastic_policy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as PolicyInput);
    let name = &input.name;

    if !input.low.is_finite()
        || !input.high.is_finite()
        || input.low < 0.0
        || input.high > 1.0
        || input.low >= input.high
    {
        return syn::Error::new(
            name.span(),
            "elastic_policy!: hysteresis must satisfy 0 <= low < high <= 1",
        )
        .to_compile_error()
        .into();
    }

    let hard_names: Vec<&syn::Ident> = input.hard.iter().map(|(name, _)| name).collect();
    let hard_values: Vec<&syn::LitStr> = input.hard.iter().map(|(_, value)| value).collect();
    let objectives: Vec<&syn::LitStr> = input.objectives.iter().collect();
    let high = input.high;
    let low = input.low;
    let predictive = input.predictive;
    let transactional = input.transactional;
    let hard_count = input.hard.len();
    let objective_count = input.objectives.len();

    quote! {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct #name {
            #(pub #hard_names: &'static str,)*
            pub objectives: &'static [&'static str],
            pub hysteresis_high: f64,
            pub hysteresis_low: f64,
            pub predictive: bool,
            pub transactional: bool,
        }

        impl #name {
            pub const fn build() -> Self {
                Self {
                    #(#hard_names: #hard_values,)*
                    objectives: &[#(#objectives),*],
                    hysteresis_high: #high,
                    hysteresis_low: #low,
                    predictive: #predictive,
                    transactional: #transactional,
                }
            }

            pub const fn hard_len(&self) -> usize {
                #hard_count
            }

            pub const fn objective_len(&self) -> usize {
                #objective_count
            }
        }

        impl Default for #name {
            fn default() -> Self {
                Self::build()
            }
        }
    }
    .into()
}

// ---------------------------------------------------------------------------
// elastic_target!
// ---------------------------------------------------------------------------

struct TargetInput {
    objective: syn::Expr,
    constraints: Vec<syn::Expr>,
}

impl Parse for TargetInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let maximize: syn::Ident = input.parse()?;
        if maximize != "maximize" {
            return Err(syn::Error::new(maximize.span(), "expected `maximize`"));
        }
        let objective: syn::Expr = input.parse()?;
        let _: Token![,] = input.parse()?;
        let subject_to: syn::Ident = input.parse()?;
        if subject_to != "subject_to" {
            return Err(syn::Error::new(subject_to.span(), "expected `subject_to`"));
        }
        let content;
        braced!(content in input);
        let constraints: Punctuated<syn::Expr, Token![,]> =
            content.parse_terminated(syn::Expr::parse, Token![,])?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after `subject_to` block"));
        }
        Ok(Self {
            objective,
            constraints: constraints.into_iter().collect(),
        })
    }
}

/// Evaluate an optimization target at the call site.
///
/// The objective is converted to `f64`; each `subject_to` expression must be a
/// boolean. The returned value exposes `objective`, `constraints`,
/// `feasible()` and `violations()` so callers can feed actual results into a
/// controller/optimizer instead of carrying decorative syntax.
#[proc_macro]
pub fn elastic_target(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TargetInput);
    let objective = &input.objective;
    let constraints = &input.constraints;
    let count = constraints.len();

    quote! {{
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct ElasticTargetValue<const N: usize> {
            pub objective: f64,
            pub constraints: [bool; N],
        }

        impl<const N: usize> ElasticTargetValue<N> {
            pub fn feasible(&self) -> bool {
                self.constraints.iter().all(|constraint| *constraint)
            }

            pub fn violations(&self) -> usize {
                self.constraints
                    .iter()
                    .filter(|constraint| !**constraint)
                    .count()
            }
        }

        ElasticTargetValue::<#count> {
            objective: (#objective) as f64,
            constraints: [#((#constraints)),*],
        }
    }}
    .into()
}
