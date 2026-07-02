# SLHAv2 — Commercial License Agreement (Draft Template)

> **AVERTISSEMENT — PROJET DE DOCUMENT (FR)**
>
> Le présent document est un **PROJET de modèle de contrat** préparé **sans
> l'intervention d'un avocat**. Il ne constitue **pas un conseil juridique**.
> Il doit être **revu, adapté et validé par un conseil juridique qualifié**
> dans chaque juridiction concernée (notamment Union européenne / France et
> États-Unis) **avant toute utilisation**. Ce projet **ne crée aucun droit ni
> aucune obligation** tant qu'un contrat définitif n'a pas été négocié et
> **signé par les deux parties**. Les mentions entre [CROCHETS] signalent des
> choix commerciaux ou juridiques restant à arbitrer.
>
> **NOTICE — DRAFT DOCUMENT (EN)**
>
> This document is a **DRAFT contract template** prepared **without the
> involvement of an attorney**. It is **not legal advice**. It must be
> **reviewed, adapted and approved by qualified legal counsel** in each
> relevant jurisdiction (in particular the European Union / France and the
> United States) **before ANY use**. This draft **creates no rights or
> obligations** unless and until a definitive agreement has been negotiated
> and **executed by both parties**. Text in [BRACKETS] marks business or
> legal decisions still to be made.

**Covered Software:** **SLHAv2** — the CCOS tile kernel, as maintained in the
repository <https://github.com/CHECKUPAUTO/SLHAv2>, including its workspace
components (`scirust`, `slha-c`, `slha-python`, `slha-mcp`) and the versions
specified in the applicable Order Form. The companion CCOS module
**TurboQuant** is covered by the parallel draft in its own repository
(`TurboQuant/legal/COMMERCIAL_LICENSE_DRAFT.md`); [a single Order Form may
reference both agreements for a combined CCOS deployment].
---

## Commercial Software License Agreement (Draft Template)

This Commercial Software License Agreement (this "**Agreement**") is entered
into as of the Effective Date by and between:

- **Licensor:** Tarek Zekriti, [acting in a personal capacity / on behalf of
  a legal entity to be identified by counsel], [address to be inserted],
  contact: contact@checkupauto.fr ("**Licensor**"); and
- **Licensee:** [full legal name], a [form of entity] organized under the laws
  of [jurisdiction], with registered office at [address], registration number
  [number] ("**Licensee**").

Each a "**Party**" and together the "**Parties**". This Agreement is the
framework for commercial licensing of the Licensed Software and takes
commercial effect only through one or more Order Forms executed by both
Parties, into which this Agreement is incorporated by reference.

### Recitals

A. Licensor makes the Licensed Software publicly available free of charge for
noncommercial purposes under the PolyForm Noncommercial License 1.0.0 (the
"**Noncommercial License**", see `LICENSE.md` in the repository), and offers
commercial licenses **exclusively for CCOS Deployments**, as described in the
repository's `LICENSING.md`. Standalone commercial licensing outside CCOS is
not offered.

B. Licensee wishes to use the Licensed Software for commercial purposes as
part of one or more CCOS Deployments, and Licensor is willing to grant such a
license on the terms of this Agreement and the applicable Order Form.

### 1. Definitions

1.1 "**Affiliate**" means any entity that directly or indirectly controls, is
controlled by, or is under common control with a Party, where "control" means
ownership of more than fifty percent (50%) of the voting securities or the
equivalent power to direct management.

1.2 "**CCOS**" means the elastic KV-cache orchestration system developed by
Licensor (`scirust::ccos`), of which SLHAv2 is the tile kernel and TurboQuant
is a module (the TQ3 tile codec and the CCOS Soft-Paging correction rung).

1.3 "**CCOS Deployment**" means an installation and operation of CCOS, by or
for Licensee, in which the Licensed Software (a) is integrated as a component
of CCOS and is invoked through the CCOS orchestration layer and its documented
interfaces; (b) is used solely to provide CCOS functionality (elastic KV-cache
orchestration and its supporting kernels, codecs and correction rungs); and
(c) is not installed, exposed, offered or otherwise usable as a standalone
library, product or service separate from CCOS. The number, identity and
boundaries of the permitted CCOS Deployments (e.g., [instances / nodes /
clusters / environments / sites]) are specified in the applicable Order Form.

1.4 "**Documentation**" means the user and technical documentation for the
Licensed Software published in the repository (including `README.md` and the
`docs/` directory) or otherwise provided by Licensor.

1.5 "**Effective Date**" means the date of the last signature on the initial
Order Form, unless that Order Form states a different date.

1.6 "**Intellectual Property Rights**" means all copyright and related rights,
patent rights, trademark rights, design rights, database rights, trade
secrets, and all other intellectual or industrial property rights, whether
registered or not, anywhere in the world.

1.7 "**Licensed Software**" means the software identified in the "Covered
Software" header of this Agreement, in the version(s) and distribution form(s)
specified in the applicable Order Form, together with any updates or new
versions that Licensor elects to make available to Licensee under this
Agreement, and the associated Documentation. For the avoidance of doubt,
versions published before the dual-license change remain available under
their original MIT OR Apache-2.0 terms and are not governed by this
Agreement.

1.8 "**Modifications**" means changes to, or derivative works of, the
Licensed Software made by or for Licensee under the license in Section 2.1.

1.9 "**Order Form**" means an ordering document executed by both Parties that
references this Agreement and specifies at least: the Licensed Software and
version(s); the deployment option elected under Section 2.2; the scope of
permitted CCOS Deployments; the fees and payment terms; the initial term and
any renewal; and the governing-law alternative elected under Section 12. An
outline is provided in Exhibit A.

1.10 "**Required Notice**" means the plain-text notice line that accompanies
the Licensed Software under the Noncommercial License: `Required Notice:
Copyright 2026 Tarek Zekriti (https://github.com/CHECKUPAUTO/)`.

1.11 "**Term**" means the term of this Agreement as set out in Section 8.1.

### 2. License Grant

2.1 **Grant.** Subject to Licensee's payment of the applicable fees and its
continued compliance with this Agreement and the applicable Order Form,
Licensor grants Licensee a non-exclusive, non-transferable (except as
permitted in Section 13.1), worldwide license, for the Term, to use,
reproduce and modify the Licensed Software **solely as part of a CCOS
Deployment** within the scope specified in the Order Form.

2.2 **Deployment option.** The Order Form must elect exactly one of the
following alternatives:

> [**Option A — Internal Use.** The license in Section 2.1 is limited to use
> for the internal business operations of Licensee [and its Affiliates].
> Licensee shall not distribute or otherwise make the Licensed Software
> available to any third party.]
>
> [**Option B — Distribution to Customers.** In addition to internal use,
> Licensee may distribute the Licensed Software, in [object code / compiled]
> form only, solely as embedded in and forming an integral part of Licensee's
> CCOS-based product identified in the Order Form, to end customers bound by
> written terms that (i) restrict use of the Licensed Software to that
> CCOS-based product, (ii) are at least as protective of Licensor as this
> Agreement, and (iii) grant no standalone rights in the Licensed Software.]

2.3 **No sublicensing.** Licensee shall not sublicense the Licensed Software,
except [if Option B is elected: to the limited extent necessary for end
customers to use the Licensed Software as embedded in the CCOS-based product,
under the end-customer terms described in Section 2.2].

2.4 **Affiliates; contractors.** [Use by Affiliates, and by contractors
acting on Licensee's behalf and for its benefit under written confidentiality
and use restrictions, is permitted within the scope of the Order Form;
Licensee remains fully responsible for their compliance.]

2.5 **Reservation of rights; Noncommercial License unaffected.** All rights
not expressly granted are reserved by Licensor. Nothing in this Agreement
limits or conditions any rights Licensee may separately hold under the
Noncommercial License for noncommercial purposes.

### 3. Restrictions

Except as expressly permitted by this Agreement or the Order Form, Licensee
shall not, and shall not permit any third party to:

(a) commercialize, market, distribute, or make available the Licensed
Software on a standalone basis, or as part of any product or service other
than a CCOS Deployment — including as a general-purpose compression,
quantization or attention library, or as a hosted or managed service exposing
the Licensed Software's functionality separately from CCOS;

(b) relicense the Licensed Software, or subject the Licensed Software or any
Modification to license terms that would require disclosure or licensing of
the Licensed Software to third parties (including copyleft terms);

(c) remove, alter or obscure any copyright, license or attribution notices,
including the Required Notice; where a permitted distribution combines
material licensed under this Agreement with material received under the
Noncommercial License, the Required Notice and the Noncommercial License
notices applicable to the latter must be preserved;

(d) use Licensor's names, trademarks, logos or product names, except for
factual attribution required by the notices; no trademark rights are granted
under this Agreement;

(e) exceed the scope (deployments, versions, deployment option) specified in
the applicable Order Form; or

(f) use the Licensed Software in violation of applicable law.

### 4. Fees, Payment and Taxes

4.1 **Fees.** Licensee shall pay the fees set out in the Order Form
[license fee / annual subscription / per-deployment fee — pricing model and
amounts to be determined]. Except as expressly stated, fees are
non-refundable.

4.2 **Payment.** Invoices are payable within [thirty (30)] days of the
invoice date. Late amounts bear interest at [the statutory rate applicable to
commercial transactions in the elected governing jurisdiction — e.g., in
France, the rate under Article L.441-10 of the Code de commerce plus the
statutory EUR 40 recovery indemnity per invoice; in the United States,
[1.5]% per month or the maximum rate permitted by law, whichever is lower].

4.3 **Taxes — EU.** All fees are exclusive of VAT and any similar indirect
taxes. Where the reverse-charge mechanism applies (Article 196 of Council
Directive 2006/112/EC), Licensee shall self-account for VAT in its Member
State and provide a valid VAT identification number.

4.4 **Taxes — US.** Licensee is responsible for all sales, use and similar
transaction taxes arising from this Agreement, excluding taxes based on
Licensor's net income. [If withholding tax applies, amounts payable shall be
grossed up so that Licensor receives the full invoiced amount — for counsel
and tax-adviser review.]

### 5. Intellectual Property; Feedback; Contributions

5.1 **Ownership.** The Licensed Software and all Intellectual Property Rights
in it are and remain the exclusive property of Licensor. Licensee acquires
only the limited license rights expressly granted in this Agreement.

5.2 **Modifications.** [As between the Parties, Licensee owns the incremental
Intellectual Property Rights in Modifications it creates, subject to
Licensor's ownership of the underlying Licensed Software; Modifications may
be used only as part of a CCOS Deployment under this Agreement and are
subject to Sections 3 and 8.3.] [Alternative — assignment or license-back of
Modifications to Licensor: for counsel review.]

5.3 **Feedback.** Licensee grants Licensor a perpetual, irrevocable,
worldwide, royalty-free, sublicensable license to use, for any purpose, any
feedback, suggestions or improvement ideas that Licensee voluntarily provides
regarding the Licensed Software.

5.4 **Contributions; CLA.** Contributions submitted by Licensee or its
personnel to the public repositories of the Licensed Software are governed by
the project's Contributor License Agreement (see `LICENSING.md`, section 5),
under which contributions are licensed to Licensor for use under **both** the
Noncommercial License and Licensor's commercial licenses (including this
Agreement). Contributions that Licensor incorporates into the Licensed
Software form part of the Licensed Software licensed to Licensee hereunder,
at no additional fee.

### 6. Warranties; Disclaimers

6.1 **Authority.** Each Party represents and warrants that it has the full
right and authority to enter into and perform this Agreement.

6.2 **[Optional limited performance warranty.]** [Licensor warrants that, for
a period of [ninety (90)] days from initial delivery, the Licensed Software
will materially conform to the Documentation. Licensee's exclusive remedy,
and Licensor's sole liability, for breach of this warranty are, at Licensor's
option: (a) repair or replacement of the non-conforming Licensed Software; or
(b) termination of the affected Order Form and refund of the fees paid for
the non-conforming Licensed Software [pro-rated for any unused subscription
period]. This warranty does not apply to Modifications not made by Licensor,
to misuse, or to use outside a CCOS Deployment.] [Alternative: no performance
warranty — the Licensed Software is provided "AS IS".]

6.3 **Disclaimer (US-style).** EXCEPT AS EXPRESSLY STATED IN THIS SECTION 6,
THE LICENSED SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS, AND LICENSOR
DISCLAIMS ALL OTHER WARRANTIES AND CONDITIONS, WHETHER EXPRESS, IMPLIED,
STATUTORY OR OTHERWISE, INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT, AND ANY
WARRANTIES ARISING FROM COURSE OF DEALING OR USAGE OF TRADE. LICENSOR DOES
NOT WARRANT THAT THE LICENSED SOFTWARE WILL BE ERROR-FREE OR THAT ITS
OPERATION WILL BE UNINTERRUPTED.

6.4 **EU statutory rights; B2B framing.** This Agreement is offered
exclusively on a business-to-business basis; Licensee represents that it acts
as a professional, for purposes within its trade, business or profession, and
not as a consumer. Nothing in this Agreement excludes or limits any warranty,
right or liability that cannot lawfully be excluded or limited under
applicable mandatory law; where and to the extent such mandatory rules apply
between the Parties (for example, statutory guarantees such as the French
*garantie légale des vices cachés*, Articles 1641 et seq. of the French Civil
Code — [the validity of its exclusion between professionals is to be assessed
by counsel]), those rules prevail, and the disclaimers in Section 6.3 apply
only to the maximum extent permitted. If either Party were nonetheless deemed
a consumer under applicable law, that Party's mandatory consumer rights
remain unaffected.

### 7. Limitation of Liability

7.1 **No indirect damages.** To the maximum extent permitted by applicable
law, neither Party shall be liable for any indirect, incidental, special,
consequential, exemplary or punitive damages, or for any loss of profits,
revenue, data, goodwill or business opportunity, arising out of or related to
this Agreement, even if advised of the possibility of such damages.

7.2 **Cap.** To the maximum extent permitted by applicable law, each Party's
total aggregate liability arising out of or related to this Agreement shall
not exceed [the total fees paid or payable by Licensee under the applicable
Order Form during the twelve (12) months preceding the event giving rise to
the claim] [alternative: a fixed amount of [amount]].

7.3 **Exclusions from the limitations.** Sections 7.1 and 7.2 do not apply
to: (a) death or personal injury caused by a Party's negligence; (b) fraud or
fraudulent misrepresentation (*dol*); (c) gross negligence or willful
misconduct (*faute lourde ou faute dolosive*; *Vorsatz oder grobe
Fahrlässigkeit*); (d) any other liability that cannot be excluded or limited
under applicable law; [(e) Licensee's payment obligations; (f) Licensee's
breach of the license scope (Sections 2 and 3) — for counsel review].

7.4 **Allocation of risk.** The Parties acknowledge that the fees reflect
this allocation of risk and that the limitations in this Section 7 are an
essential basis of the bargain between them; to the extent permitted by
applicable law, they apply even if a limited remedy fails of its essential
purpose.

### 8. Term and Termination

8.1 **Term.** This Agreement takes effect on the Effective Date and remains
in force [for as long as an Order Form is in effect]. Each Order Form runs
for [an initial term of [twelve (12)] months, renewing for successive
[twelve (12)]-month periods unless either Party gives [sixty (60)] days'
notice of non-renewal] [alternative: a perpetual license limited to the
version(s) delivered, with optional maintenance].

8.2 **Termination for cause.** Either Party may terminate this Agreement or
an affected Order Form by written notice if the other Party materially
breaches it and fails to cure the breach within **thirty (30) days** of
written notice describing the breach, or, to the extent permitted by
applicable insolvency law, upon the other Party's insolvency, bankruptcy or
cessation of business.

8.3 **Effect of termination.** Upon expiry or termination: (a) the licenses
granted under Section 2 end and Licensee shall cease all use of the Licensed
Software under this Agreement; (b) Licensee shall delete or destroy its
copies of the Licensed Software and Modifications held under this Agreement,
except one archival copy and copies required to be retained by law, and shall
certify deletion upon request; (c) termination does not affect (i) any rights
Licensee holds under the Noncommercial License for noncommercial purposes,
(ii) [if Option B was elected: rights of end customers in copies of the
CCOS-based product properly distributed before termination — for counsel
review], or (iii) rights and obligations accrued before termination.

8.4 **Survival.** Sections 1, 3, 4 (for accrued amounts), 5, 6.3, 6.4, 7,
8.3, 8.4, 9, 10, 11 (for [two (2)] years after termination), 12 and 13
survive expiry or termination of this Agreement.

### 9. Compliance: Export, Sanctions, Anti-Corruption

9.1 **Export controls.** Each Party shall comply with applicable export
control laws, including Regulation (EU) 2021/821 on dual-use items and the
U.S. Export Administration Regulations (EAR), and shall not export, re-export
or transfer the Licensed Software in violation of such laws. [Export
classification of the Licensed Software to be confirmed.]

9.2 **Sanctions.** Licensee represents that it is not, and is not owned or
controlled by, a person listed on applicable sanctions lists (including EU
restrictive measures and U.S. OFAC lists), and shall not use or make the
Licensed Software available in embargoed or comprehensively sanctioned
territories.

9.3 **Anti-corruption.** Each Party shall comply with applicable
anti-corruption laws, including the French Sapin II law, the U.S. Foreign
Corrupt Practices Act and the UK Bribery Act 2010.

### 10. Data Protection

10.1 The Licensed Software does not, by itself, collect, transmit or process
personal data on behalf of Licensor, and the license granted hereunder does
not involve any processing of personal data by Licensor for Licensee.

10.2 Business contact details exchanged for the administration of this
Agreement are processed by each Party as an independent controller, in
accordance with applicable data protection law (including, where applicable,
Regulation (EU) 2016/679 — GDPR).

10.3 [If support, maintenance or professional services under an Order Form
involve access by Licensor to personal data processed by Licensee, the
Parties shall first execute a data processing agreement meeting the
requirements of Article 28 GDPR (and, where relevant, applicable U.S. state
privacy laws).]

### 11. Audit

Upon at least thirty (30) days' prior written notice, and no more than once
in any twelve (12) month period, Licensor (or an independent auditor bound by
confidentiality) may audit Licensee's records and systems relevant to its use
of the Licensed Software, during normal business hours and without
unreasonable disruption to Licensee's operations, solely to verify compliance
with this Agreement and the Order Form. Information obtained is confidential
and may be used only for compliance purposes. Licensee shall promptly pay any
identified shortfall; if the underpayment exceeds five percent (5%) of the
amounts due for the audited period, Licensee shall also bear the reasonable
costs of the audit.

### 12. Governing Law and Jurisdiction

12.1 The Order Form must elect **exactly one** of the following alternatives.
The choice must be made deliberately, per Order Form, and **reviewed by
qualified counsel of both Parties** before execution:

> [**Alternative 1 — France.** This Agreement is governed by French law. Any
> dispute arising out of or in connection with this Agreement shall be
> subject to the exclusive jurisdiction of the **Tribunal de commerce de
> Paris** (Paris Commercial Court), including for interim and summary
> proceedings, notwithstanding plurality of defendants or third-party
> claims.]
>
> [**Alternative 2 — United States (Delaware).** This Agreement is governed
> by the laws of the State of Delaware, without regard to its conflict-of-law
> rules. The state and federal courts located in Wilmington, Delaware shall
> have exclusive jurisdiction and venue, and each Party consents to personal
> jurisdiction there. [Jury-trial waiver — for counsel review.]]

12.2 The United Nations Convention on Contracts for the International Sale of
Goods (CISG) does not apply to this Agreement. [The Uniform Computer
Information Transactions Act (UCITA) is excluded where enacted.]

### 13. Miscellaneous

13.1 **Assignment.** Neither Party may assign this Agreement without the
other Party's prior written consent, not to be unreasonably withheld,
[except to an Affiliate or in connection with a merger, reorganization or
sale of substantially all relevant assets, upon written notice, provided the
assignee is not a direct competitor of the other Party].

13.2 **Force majeure.** Neither Party is liable for failure or delay caused
by events beyond its reasonable control, provided it notifies the other Party
and resumes performance as soon as reasonably possible; payment obligations
for amounts already due are not excused.

13.3 **Notices.** Notices under this Agreement must be in writing, in English
[or French], and sent: to Licensor at contact@checkupauto.fr [postal address
to be inserted]; to Licensee at the addresses stated in the Order Form.
Notices are deemed received [upon written acknowledgment, or one (1) business
day after transmission by email without a delivery failure].

13.4 **Entire agreement; precedence.** This Agreement, together with each
Order Form and any exhibits, is the entire agreement between the Parties
regarding its subject matter and supersedes all prior discussions. In case of
conflict, an Order Form prevails over this Agreement for the transaction it
covers. The Noncommercial License remains a separate, independent public
license and is not modified by this Agreement.

13.5 **Severability.** If any provision is held invalid or unenforceable, it
shall be enforced to the maximum extent permissible and the remainder of the
Agreement remains in effect.

13.6 **No waiver.** Failure to enforce a provision is not a waiver of the
right to enforce it later. Waivers must be in writing.

13.7 **Independent contractors.** The Parties are independent contractors;
this Agreement creates no partnership, agency or joint venture.

13.8 **Counterparts; electronic signature.** This Agreement and Order Forms
may be executed in counterparts and by electronic signature, which shall have
the same effect as handwritten signatures to the extent permitted by
applicable law (including Regulation (EU) No 910/2014 (eIDAS) and the U.S.
ESIGN Act / UETA).

### Signature Blocks

**[DO NOT SIGN THIS DRAFT.** This template has no legal effect and must not
be executed in its current form. Signature blocks are to be completed only in
the final, counsel-reviewed version and on the applicable Order Form.]

| Licensor | Licensee |
| --- | --- |
| Name: Tarek Zekriti | Name: [name] |
| Title: [title / capacity] | Title: [title] |
| Date: [date] | Date: [date] |
| Signature: ______________ | Signature: ______________ |

### Exhibit A — Order Form (Outline, To Be Completed)

Each Order Form referencing this Agreement should specify at least:

1. Parties, addresses, and (for EU Licensees) VAT identification number.
2. Licensed Software and covered version(s) / distribution form(s).
3. Deployment option elected (Section 2.2, Option A or B) and, for Option B,
   the identified CCOS-based product.
4. Scope of permitted CCOS Deployments ([instances / nodes / clusters /
   environments / sites]).
5. Fees, pricing model, payment schedule and currency.
6. Initial term, renewal and notice periods (Section 8.1).
7. Governing-law alternative elected (Section 12.1).
8. Support / maintenance terms, if any [and DPA if Section 10.3 applies].
9. Any special terms (which prevail over this Agreement per Section 13.4).
10. Signatures of both Parties.
