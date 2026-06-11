# 3. Method

We address multimodal emotion-cause pair extraction (MECPE) in conversations, where the goal is to identify, for each utterance that expresses emotion, the utterances that cause it. Our model couples a relation-aware multimodal encoder with three complementary sources of decision evidence: (i) *necessity evidence* obtained by probing how much each candidate cause and each non-verbal modality contributes to a pair prediction; (ii) a *bounded positional prior* that exploits the locality of causes without letting position dominate; and (iii) *reasoning evidence* distilled offline from a large language model (LLM). The three signals are combined by a gated fusion module and trained jointly. Notation is introduced where first used.

## 3.1 Task Formulation

A conversation is a sequence of $N$ utterances $\mathcal{D}=\{u_1,\ldots,u_N\}$. Each utterance carries textual, acoustic and visual features together with a speaker identity,
$$
u_i=\{x_i^{t},\,x_i^{a},\,x_i^{v},\,s_i\}.
$$
For an ordered pair $(u_i,u_j)$ in which $u_i$ is the candidate *emotion* utterance and $u_j$ the candidate *cause* utterance, the model predicts
$$
p_{ij}=P\!\left(y_{ij}=1 \mid u_i,u_j,\mathcal{D}\right),
$$
where $y_{ij}=1$ iff $u_j$ is a cause of the emotion expressed in $u_i$. Two auxiliary utterance-level heads predict an emotion probability $\hat p^{\,e}_i$ and a cause probability $\hat p^{\,c}_j$; these provide supervision signals and drive the candidate gating of Section 3.4.

## 3.2 Overview

The encoder (Section 3.3) maps the conversation to contextualized utterance representations and forms a representation $h_{ij}^{\mathrm{pair}}$ for every candidate pair. A soft candidate gate (Section 3.4) concentrates the more expensive computations on the most promising pairs.

For these gated pairs we obtain two complementary forms of causal-style evidence—perturbation-based necessity from the model itself (Section 3.5) and a distilled textual *counterfactual* vector from an LLM (Section 3.7)—and add a bounded positional prior (Section 3.6) at the score level. A complementarity-aware gating network (Section 3.8) fuses the pair representation with the two evidence vectors, conditioning on the model's confidence, the LLM's reliability, and their agreement so that each source compensates the other's weakness, to produce the final pair probability. Training combines the pair objective, the auxiliary emotion/cause objectives, and the evidence-related objectives (Section 3.9).

## 3.3 Relation-Aware Multimodal Encoder

**Multimodal fusion.** Each utterance is first projected into a shared $H$-dimensional space by fusing its modalities and speaker identity,
$$
h_i^{0}=\mathrm{Fuse}\!\left(x_i^{t},x_i^{a},x_i^{v},s_i\right)\in\mathbb{R}^{H}.
$$

**Conversation graph.** We build a heterogeneous graph over utterances with four relation types $\mathcal{R}=\{\textit{self-loop},\textit{same-speaker},\textit{reply/adjacent},\textit{emotion-transition}\}$:

- *self-loop*: $i=j$;
- *same-speaker*: $s_i=s_j$ and $i\neq j$;
- *reply/adjacent*: $|i-j|=1$, or $u_j$ is the reply target of $u_i$ (with a fallback to the nearest preceding cross-speaker utterance); adjacency and reply share a single relation type;
- *emotion-transition*: among structurally connected turns, those whose emotion labels differ are re-typed as emotion-transition, which takes priority over the structural off-diagonal types.

The emotion-transition relation depends on emotion labels, which are gold during training and predicted at inference. To avoid a train–inference mismatch, the source label of each such edge is drawn by scheduled sampling,
$$
\ell_i=
\begin{cases}
\ell_i^{\mathrm{gold}}, & \text{with probability } \varepsilon_t,\\
\ell_i^{\mathrm{pred}}, & \text{with probability } 1-\varepsilon_t,
\end{cases}
\qquad \varepsilon_t:\,1\rightarrow 0,
$$
where $\ell_i^{\mathrm{pred}}$ is the detached arg-max prediction of a preliminary emotion head. As $\varepsilon_t$ anneals to $0$, the training-time graph converges to the inference-time graph. This relation, together with the evidence modules of Sections 3.5 and 3.7, is activated only after a warmup of $K$ epochs; during warmup the graph uses only the three structural relation types.

**Relation-aware propagation.** Messages are aggregated with relation-specific attention,
$$
\alpha_{ij}^{r}=\operatorname*{softmax}_{j}\!\Big(\mathrm{LeakyReLU}\big(a_r^{\top}[\,W_r h_i^{(l)};\,W_r h_j^{(l)}\,]\big)\Big),
\qquad
h_i^{(l+1)}=\sigma\!\Big(\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_i^{r}}\alpha_{ij}^{r}\,W_r h_j^{(l)}\Big).
$$
After $L$ layers we obtain contextualized representations $\tilde h_i\in\mathbb{R}^{H}$. We write the encoder compactly as $\tilde H=\mathrm{Enc}\big(\{h_i^{0}\}\big)$, a form reused in Section 3.5. The representation of a candidate pair concatenates the two utterance vectors with their element-wise product and absolute difference,
$$
h_{ij}^{\mathrm{pair}}=\big[\tilde h_i;\,\tilde h_j;\,\tilde h_i\odot\tilde h_j;\,|\tilde h_i-\tilde h_j|\big]\in\mathbb{R}^{4H}.
$$
**Self-contained pairs.** For a self-cause pair ($j=i$) this representation degenerates: the difference block $|\tilde h_i-\tilde h_i|$ is identically zero and the product collapses to $\tilde h_i\odot\tilde h_i$, so a quarter of $h_{ii}^{\mathrm{pair}}$ carries no signal—an avoidable handicap given that such pairs are about half of the labels (Section 3.7). We therefore add a single learnable self-loop signature $e^{\mathrm{self}}\in\mathbb{R}^{H}$ into the otherwise-dead difference block on the diagonal,
$$
h_{ii}^{\mathrm{pair}}=\big[\tilde h_i;\,\tilde h_i;\,\tilde h_i\odot\tilde h_i;\,e^{\mathrm{self}}\big],
$$
which gives self-cause pairs a distinct, learnable marker for the scoring, distillation, and fusion heads while leaving the $4H$ dimension and every cross-utterance pair ($j\neq i$) unchanged; the dedicated counterfactual treatment of these self-cause pairs at the LLM-evidence stage is described in Section 3.7. The same signature is applied identically in the factual score and in the counterfactual re-scorings of Section 3.5, so the necessity difference $\Delta_{ij}^{u}$ remains uncontaminated by the marker; initialized at $e^{\mathrm{self}}=0$, it recovers the plain representation at the start of training.

## 3.4 Soft Candidate Gating

Enumerating all pairs is quadratic, and computing evidence for every pair is costly. We therefore define a soft relevance gate from the auxiliary heads,
$$
\pi_{ij}=\hat p^{\,e}_i\cdot \hat p^{\,c}_j\in[0,1],
$$
and allocate the evidence computations of Sections 3.5 and 3.7 to the top-$M$ pairs by $\pi_{ij}$ (where $M$ is the evidence budget, a hyperparameter), denoted $\mathcal{T}$. Crucially, gating governs *where computation is spent*, not which pairs may be positive: every valid pair still contributes to the pair loss (Section 3.9), so no candidate is permanently discarded.

## 3.5 Perturbation-Based Necessity Evidence

A candidate cause should matter only if removing it changes the prediction. We quantify this with a scalar pair-scoring head $f:\mathbb{R}^{4H}\!\to\!\mathbb{R}$, whose factual score is
$$
s_{ij}=f\!\left(h_{ij}^{\mathrm{pair}}\right).
$$
The head is supervised directly by the pair labels,
$$
\mathcal{L}_{\mathrm{score}}=\mathrm{BCE}\!\left(\sigma(s_{ij}),\,y_{ij}\right),
$$
so that the score is calibrated and the differences defined below are meaningful.

**Anchored baselines.** Perturbations replace an element with a neutral baseline rather than zero, to stay close to the training distribution. Baselines are anchored to data statistics: with detached exponential-moving-average means $\bar h$ (utterance space) and $\bar x^{m}$ (modality-$m$ input space),
$$
b^{\mathrm{utt}}=\bar h+\delta^{\mathrm{utt}},\qquad
b^{m}=\bar x^{m}+\delta^{m},\qquad
\mathcal{L}_{\mathrm{anc}}=\big\|\delta^{\mathrm{utt}}\big\|_2^2+\sum_{m}\big\|\delta^{m}\big\|_2^2,
$$
where the learnable residuals $\delta$ are penalized so the baselines remain near genuine "no-information" means.

**Cause necessity.** Replacing the contextualized cause representation $\tilde h_j$ by $b^{\mathrm{utt}}$ and re-scoring yields the necessity of the cause for the pair decision,
$$
s_{ij}^{-u}=f\!\left(\big[\tilde h_i;\,\tilde h_j^{-u};\,\tilde h_i\odot\tilde h_j^{-u};\,|\tilde h_i-\tilde h_j^{-u}|\big]\right),
\qquad
\Delta_{ij}^{u}=s_{ij}-s_{ij}^{-u}.
$$
Because the textual stream is the backbone of the encoder, this representation-level term also captures the semantic (textual) necessity of the cause; we therefore reserve modality perturbation for the non-verbal streams.

**Pair-local modality contribution.** For the acoustic and visual modalities $m\in\{a,v\}$, a faithful perturbation must propagate through the non-linear encoder. We re-encode the whole conversation once per modality with that modality replaced by its baseline at the input,
$$
\tilde H^{-m}=\mathrm{Enc}\!\left(\{h_i^{0}\}\,\big|\,x^{m}\!\leftarrow b^{m}\right),\qquad m\in\{a,v\}.
$$
This costs only two extra encodings per conversation and is shared by all of its candidate pairs. To make the modality perturbation *pair-local* and directly comparable to the cause-necessity term—which also replaces only the cause side—we form the counterfactual pair representation by keeping the original emotion representation $\tilde h_i$ and using the perturbed cause representation $\tilde h_j^{-m}$ from $\tilde H^{-m}$:
$$
\Delta_{ij}^{m}=s_{ij}-f\!\left(\big[\tilde h_i;\,\tilde h_j^{-m};\,\tilde h_i\odot\tilde h_j^{-m};\,|\tilde h_i-\tilde h_j^{-m}|\big]\right),
$$
where $\tilde h_j^{-m}$ is the $j$-th utterance vector from $\tilde H^{-m}$. This design ensures that $\Delta_{ij}^{m}$ measures how much modality $m$ in the *cause utterance* contributes to the pair score, paralleling $\Delta_{ij}^{u}$ which measures how much the cause utterance as a whole contributes. The normalized modality terms and their weights are
$$
\tilde\Delta_{ij}^{m}=\frac{\Delta_{ij}^{m}-\mu_m}{\sigma_m+\epsilon},\qquad
w_{ij}^{m}=\operatorname*{softmax}_{m\in\{a,v\}}\big(\tilde\Delta_{ij}^{m}\big),
$$
where $(\mu_m,\sigma_m)$ are running statistics that make the two modalities comparable. The cause term is standardized within its own group as $\tilde\Delta_{ij}^{u}=(\Delta_{ij}^{u}-\mu_u)/(\sigma_u+\epsilon)$.

**Necessity-evidence vector.** Collecting the representation-level cause term and the pair-local modality terms gives
$$
z_{ij}^{\mathrm{nec}}=\big[\,\tilde\Delta_{ij}^{u};\ \tilde\Delta_{ij}^{a};\ \tilde\Delta_{ij}^{v};\ w_{ij}^{a};\ w_{ij}^{v}\,\big]\in\mathbb{R}^{5}.
$$
This vector is consumed by the fusion module (Section 3.8) with a stop-gradient, so that noisy early-training evidence informs the classifier without destabilizing the backbone.

**Necessity calibration.** Although $\mathcal{L}_{\mathrm{score}}$ calibrates the factual score $s_{ij}$, it provides no direct supervision for the perturbation differences $\Delta_{ij}^{u}$. To ensure that the necessity evidence is semantically meaningful—that removing a true cause decreases the score and removing a non-cause does not—we add a hinge-style calibration loss,
$$
\mathcal{L}_{\mathrm{cal}}=\frac{1}{|\mathcal{B}|}\sum_{(i,j)\in\mathcal{B}}\Big[\mathbb{1}[y_{ij}=1]\,\mathrm{ReLU}\!\big(-\Delta_{ij}^{u}\big)+\mathbb{1}[y_{ij}=0]\,\mathrm{ReLU}\!\big(\Delta_{ij}^{u}-\kappa_{\mathrm{cal}}\big)\Big],
$$
where $\kappa_{\mathrm{cal}}\ge 0$ is a small margin. For positive pairs the loss penalizes $\Delta_{ij}^{u}\le 0$ (cause removal should decrease the score); for negative pairs it penalizes $\Delta_{ij}^{u}>\kappa_{\mathrm{cal}}$ (non-cause removal should not increase the score beyond a margin). This direct supervision anchors the necessity evidence to the pair labels, complementing the indirect calibration from $\mathcal{L}_{\mathrm{score}}$.

We stress that this necessity evidence is a *model-level* quantity: it measures the sensitivity of the model's own prediction to controlled perturbations of its inputs, and is used to regularize and explain predictions rather than to assert causal relations in the world.

## 3.6 Bounded Positional Prior

Causes tend to lie near the emotion utterance, so relative position is informative; the risk is over-reliance, not the information itself. We encode position as a bounded additive prior on the pair score. With normalized signed distance $d_{ij}=(i-j)/N\in[-1,1]$ and a small network $\psi$,
$$
b_{ij}^{\mathrm{pos}}=\eta\cdot\tanh\!\big(\psi(d_{ij})\big)\in(-\eta,\eta),
\qquad
\mathcal{L}_{\mathrm{pos}}=\frac{1}{|\mathcal{B}|}\sum_{(i,j)\in\mathcal{B}}\big(b_{ij}^{\mathrm{pos}}\big)^2 .
$$
The amplitude $\eta$ caps the prior and $\mathcal{L}_{\mathrm{pos}}$ further discourages it from dominating, so position can refine but not override the semantic evidence. The prior enters the final logit in Section 3.8.

## 3.7 LLM-Guided Counterfactual Evidence Distillation

LLMs reason effectively over text but cannot perceive raw acoustic or visual signals; we therefore restrict their role to text-grounded reasoning and obtain modality evidence solely from Section 3.5. Querying an LLM at inference is also impractical, so its reasoning is distilled offline into a lightweight student. The necessity evidence of Section 3.5 is a *model-internal* perturbation signal—precise about what the current model relies on, but semantically shallow and noisy early in training. We therefore ask the LLM for a deliberately *complementary* signal: a textual **counterfactual** judgement of the cause–effect relation, which is semantically informed but only weakly supervised. The two are reconciled by the fusion of Section 3.8.

**Counterfactual reasoning evidence.** For a candidate pair, a fixed structured prompt asks the LLM to reason counterfactually about an intervention on the candidate cause and to return four calibrated scalars in $[0,1]$:
$s_{ij}^{\mathrm{nec}}$ (*necessity*), how likely the emotion in $u_i$ would **disappear or weaken** had $u_j$ not occurred (or been neutral);
$s_{ij}^{\mathrm{suf}}$ (*sufficiency*), how likely $u_j$ **alone** suffices to elicit that emotion;
$s_{ij}^{\mathrm{dir}}$ (*direction*), whether $u_j$ causes $u_i$ ($1$) rather than the reverse or a mere reaction ($0$);
and $s_{ij}^{\mathrm{spur}}$ (*spuriousness*), the probability that $u_i,u_j$ merely co-occur or share a topic **without** a causal link. Reliability is measured by the inter-sample agreement $\rho_{ij}\in[0,1]$ across $k$ stochastic LLM samples. The reasoning-evidence vector is
$$
z_{ij}^{\mathrm{rea}}=\big[\,s_{ij}^{\mathrm{nec}};\ s_{ij}^{\mathrm{suf}};\ s_{ij}^{\mathrm{dir}};\ s_{ij}^{\mathrm{spur}};\ \rho_{ij}\,\big]\in[0,1]^{5}.
$$
We stress that these are *textual* counterfactual judgements obtained by prompting, not a formal structural-causal-model identification; they provide weak external supervision that is complementary to—rather than a substitute for—the perturbation evidence $z_{ij}^{\mathrm{nec}}$.

**Self-contained causes.** A substantial share of emotions are triggered not by another turn but by the event described *within the emotion utterance itself* (a self-contained cause, $j=i$); in this corpus such self-cause pairs account for roughly half of the gold labels, which is also why their pair representation already receives the dedicated self-loop signature $e^{\mathrm{self}}$ of Section 3.3. For these pairs the "remove $u_j$" framing is ill-posed, since deleting the utterance also deletes the emotion it carries, and a single counterfactual phrasing would systematically under-score them. We therefore branch the prompt on whether $j=i$. For $j\neq i$ the four scalars retain the cross-utterance reading above. For $j=i$ the counterfactual is taken over the **event described** in $u_i$ rather than over the sentence: $s_{ii}^{\mathrm{nec}}$ asks how much the emotion would weaken had that event not happened, $s_{ii}^{\mathrm{suf}}$ whether the utterance's own content alone elicits the emotion, $s_{ii}^{\mathrm{dir}}$ whether the emotion arises from this utterance's own content (versus being a reaction to another turn), and $s_{ii}^{\mathrm{spur}}$ the probability that the utterance merely carries the emotion while its stated content is not the trigger. Both branches share the same four-scalar schema, so the output and all downstream consumers are unchanged. Each branch is anchored by a small set of fixed few-shot exemplars (a self-contained trigger as a positive and a pure reaction as a negative for the self-cause branch). Given their prevalence, the self-pair $(i,i)$ is always included among the annotation candidates for every non-neutral emotion utterance, independently of the locality window used for cross-utterance candidates.

**Offline annotation.** Annotation is performed once with a frozen base model $\theta_0$ (the model after warmup). The quantities used to select hard pairs are read from $\theta_0$ and then fixed, so labels do not drift during training. A pair is *hard* when the model is unconfident or its decision conflicts with the necessity evidence,
$$
\mathcal{P}_{\mathrm{hard}}=\Big\{(i,j)\ \Big|\ \max(p_{ij},1-p_{ij})<\tau\ \ \text{or}\ \ \mathrm{Conflict}_{ij}=1\Big\},
$$
$$
\mathrm{Conflict}_{ij}=\mathbb{1}\Big[(p_{ij}\!\ge\!0.5\wedge \Delta_{ij}^{u}\!\le\!0)\ \vee\ (p_{ij}\!<\!0.5\wedge \Delta_{ij}^{u}\!>\!\kappa)\Big].
$$
Here $\kappa$ is a small conflict margin, distinct from the calibration margin $\kappa_{\mathrm{cal}}$ of Section 3.5. To match the distribution the student sees at inference, the annotated set is drawn from the same population $\mathcal{T}$ used for evidence and stratified into hard and easy pairs,
$$
\mathcal{S}=\big(\mathcal{T}\cap\mathcal{P}_{\mathrm{hard}}\big)\ \cup\ \mathrm{sample}\big(\mathcal{T}\setminus\mathcal{P}_{\mathrm{hard}}\big).
$$

**Distillation.** A student head predicts the reasoning evidence from the pair representation alone,
$$
\hat z_{ij}^{\mathrm{rea}}=q\!\left(h_{ij}^{\mathrm{pair}}\right),\qquad q:\mathbb{R}^{4H}\to[0,1]^{5},
$$
which prevents it from trivially copying the necessity evidence. Since all targets share the $[0,1]$ range, distillation uses a reliability-weighted grouped binary cross-entropy,
$$
\mathcal{L}_{\mathrm{dst}}=\frac{1}{|\mathcal{S}|}\sum_{(i,j)\in\mathcal{S}}\rho_{ij}\sum_{c=1}^{5}\mathrm{BCE}\!\left(\hat z_{ij,c}^{\mathrm{rea}},\,z_{ij,c}^{\mathrm{rea}}\right).
$$
At inference the LLM is never queried; the student supplies $\hat z_{ij}^{\mathrm{rea}}$, whose fifth coordinate $\hat\rho_{ij}=\hat z_{ij,5}^{\mathrm{rea}}$ serves as an inference-time reliability estimate consumed by the fusion module. To ensure the student is already partially trained when the evidence pathway activates, the distillation loss is applied from the very first epoch (whenever annotations are available), even though the student's output does not yet enter fusion during warmup (Section 3.9).

## 3.8 Complementarity-Aware Evidence Fusion and Prediction

Fusion is designed to realize an explicit complementarity between the two sources rather than a generic blend: the LLM should **compensate** the small model where the latter is uncertain, while the small model should **lead** where the LLM is unreliable or disagrees with the model's own necessity evidence. We therefore condition the fusion on confidence, reliability, and the agreement between the two views.

For pairs in $\mathcal{T}$ a smooth presence weight derived from the gate indicates how firmly a pair lies within the evidence budget,
$$
g^{\mathrm{ev}}_{ij}=\sigma\!\big(s_\pi(\pi_{ij}-\pi_M)\big)\in[0,1],\qquad a_{ij}=g^{\mathrm{ev}}_{ij},
$$
with $\pi_M$ the $M$-th largest relevance and $s_\pi$ a temperature; $a_{ij}$ varies continuously across the budget boundary and pairs outside $\mathcal{T}$ approach $a_{ij}\!\to\!0$.

**Reliability scaling.** The distilled evidence is attenuated by its own predicted reliability, so an uncertain teacher is automatically down-weighted (the small model leads). To avoid distorting the reliability coordinate itself, we scale only the first four components and preserve $\hat\rho_{ij}$ in place:
$$
\bar z_{ij}^{\mathrm{rea}}=\big[\,\hat\rho_{ij}\,\hat z_{ij,1:4}^{\mathrm{rea}};\ \hat\rho_{ij}\,\big].
$$
This ensures that the $\phi$ network and the gate receive an undistorted reliability estimate rather than $\hat\rho_{ij}^2$.

**Confidence and agreement.** We read the small model's confidence from its factual pair score $s_{ij}$ (Section 3.5). Although the final logit also includes the positional prior and the agreement term, $s_{ij}$ serves as a stable *prior-confidence* proxy that avoids circular dependency with the fusion output; the learnable gate parameters can compensate for any systematic discrepancy. Confidence and agreement are
$$
c_{ij}=\big|2\sigma(s_{ij})-1\big|\in[0,1],\qquad
\alpha_{ij}^{\mathrm{agr}}=1-\big|\sigma(\tilde\Delta_{ij}^{u})-s_{ij}^{\mathrm{nec}}\big|\in[0,1],\qquad
\mathrm{cf}_{ij}=1-\alpha_{ij}^{\mathrm{agr}},
$$
where $\alpha_{ij}^{\mathrm{agr}}$ is the necessity **agreement** and $\mathrm{cf}_{ij}$ the corresponding conflict; the map $\sigma(\cdot)$ brings the standardized necessity $\tilde\Delta_{ij}^{u}$ into the same $[0,1]$ range as $s_{ij}^{\mathrm{nec}}$, so the model's and the LLM's necessity readings are directly comparable. The fused evidence representation projects back to the pair dimension,
$$
h_{ij}^{\mathrm{ev}}=\phi\!\left(\big[h_{ij}^{\mathrm{pair}};\,\mathrm{sg}(z_{ij}^{\mathrm{nec}});\,\bar z_{ij}^{\mathrm{rea}};\,a_{ij};\,c_{ij};\,\alpha_{ij}^{\mathrm{agr}}\big]\right),
\qquad
\phi:\mathbb{R}^{4H+5+5+3}\to\mathbb{R}^{4H},
$$
where $\mathrm{sg}(\cdot)$ is the stop-gradient operator and the input groups $4H+5+5+3$ correspond, in order, to the pair representation $h_{ij}^{\mathrm{pair}}$, the necessity vector $z_{ij}^{\mathrm{nec}}$, the reliability-scaled reasoning vector $\bar z_{ij}^{\mathrm{rea}}$, and the three scalars $\{a_{ij},c_{ij},\alpha_{ij}^{\mathrm{agr}}\}$. The balancing gate is **conditioned** on confidence, reliability, and conflict,
$$
g_{ij}=\sigma\!\big(W_g[\,h_{ij}^{\mathrm{pair}};\,h_{ij}^{\mathrm{ev}};\,c_{ij};\,\hat\rho_{ij};\,\mathrm{cf}_{ij}\,]+b_g\big),
\qquad
h_{ij}^{\mathrm{final}}=g_{ij}\odot h_{ij}^{\mathrm{pair}}+(1-g_{ij})\odot h_{ij}^{\mathrm{ev}},
$$
so a less confident small model (small $c_{ij}$) can defer to the evidence, whereas an unreliable or conflicting LLM (small $\hat\rho_{ij}$, large $\mathrm{cf}_{ij}$) is suppressed in favor of the pair representation.

**Necessity-modulated agreement.** The pair probability combines the fused representation with the bounded positional prior and an agreement term. A naive agreement boost $w_\alpha(2\alpha_{ij}^{\mathrm{agr}}-1)$ is direction-agnostic: it rewards consensus regardless of whether both sources agree the pair *is* or *is not* causal, so that "both say no" would incorrectly push the logit up. We therefore modulate the agreement by the average necessity of the two sources,
$$
\bar n_{ij}=\tfrac{1}{2}\big(\sigma(\tilde\Delta_{ij}^{u})+s_{ij}^{\mathrm{nec}}\big)\in[0,1],
$$
and define the final logit as
$$
p_{ij}=\sigma\!\big(w_f^{\top}h_{ij}^{\mathrm{final}}+b_f+b_{ij}^{\mathrm{pos}}+w_\alpha\,(2\alpha_{ij}^{\mathrm{agr}}-1)\,\bar n_{ij}\big),
$$
with a scalar $w_\alpha$ initialized at $0$. When both sources agree the pair is causal ($\bar n_{ij}\approx 1$), the agreement boost is fully active; when both agree it is not ($\bar n_{ij}\approx 0$), the boost vanishes, preventing false-positive inflation. Dropping the conditioning signals $\{c_{ij},\hat\rho_{ij},\mathrm{cf}_{ij},\alpha_{ij}^{\mathrm{agr}}\}$ (equivalently $\hat\rho_{ij}\!\equiv\!1$, $w_\alpha\!=\!0$) recovers the unconditioned gate $g_{ij}=\sigma(W_g[h_{ij}^{\mathrm{pair}};h_{ij}^{\mathrm{ev}}]+b_g)$, which we retain as an ablation baseline.

## 3.9 Training

The pair objective is a class-weighted binary cross-entropy evaluated over all positive pairs and a sampled set of negatives at ratio $r$ (within the valid upper-triangular region), which preserves recall while controlling cost and class imbalance,
$$
\mathcal{L}_{\mathrm{pair}}=\frac{1}{|\mathcal{P}^{+}\cup\mathcal{N}_r|}\sum_{(i,j)\in\mathcal{P}^{+}\cup\mathcal{N}_r}\mathrm{wBCE}_{\omega}\!\left(p_{ij},y_{ij}\right).
$$
With the auxiliary emotion and cause objectives $\mathcal{L}_{\mathrm{emo}},\mathcal{L}_{\mathrm{cau}}$ and the preliminary emotion head $\mathcal{L}_{\mathrm{emo}}^{\mathrm{pre}}$ (Section 3.3; the head that supplies the emotion-transition labels, entering with unit weight as written below), the classification loss is
$$
\mathcal{L}_{\mathrm{cls}}=\mathcal{L}_{\mathrm{pair}}+\alpha_1\mathcal{L}_{\mathrm{emo}}+\alpha_2\mathcal{L}_{\mathrm{cau}}+\mathcal{L}_{\mathrm{emo}}^{\mathrm{pre}},
$$
and the overall objective adds the score, distillation, position, anchoring, and calibration terms,
$$
\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\beta\,\mathcal{L}_{\mathrm{score}}+\lambda_1\,\mathcal{L}_{\mathrm{dst}}+\lambda_2\,\mathcal{L}_{\mathrm{pos}}+\gamma\,\mathcal{L}_{\mathrm{anc}}+\gamma\,\mathcal{L}_{\mathrm{cal}}.
$$
The calibration weight shares $\gamma$ with the anchoring regularizer since both are small stabilizing penalties.

Training proceeds in two phases. During the first $K$ warmup epochs the encoder and the auxiliary and scoring heads are trained with $\mathcal{L}_{\mathrm{cls}}$ and $\mathcal{L}_{\mathrm{score}}$, the graph uses only the three structural relation types (no emotion-transition edges), and evidence fusion is disabled by zeroing the presence weight $a_{ij}$ so that the fusion gate falls back to the pair representation alone. The student head, however, is trained from epoch 1 via $\mathcal{L}_{\mathrm{dst}}$ (whenever offline annotations are available), so that it is already partially warmed up when the evidence pathway activates. The base model is then frozen to produce the offline LLM annotations of Section 3.7. In the second phase all terms are active, the emotion-transition relation is introduced with scheduled sampling that anneals toward predicted labels, the anchored baselines and all heads are optimized jointly, and the necessity evidence enters fusion through the stop-gradient.

**Hyperparameters.** Although the method introduces several coefficients, the vast majority are fixed at grounded defaults and only a handful are tuned. We expose **four** knobs: the warmup length $K$, the evidence budget $M$, the distillation weight $\lambda_1$, and the fusion mode (the complementarity-aware gate of Section 3.8, or the unconditioned gate as an ablation). All remaining quantities are held fixed across datasets: the loss weights of the auxiliary and regularizing terms ($\alpha_1=\alpha_2=\beta=1$ for the main objectives, with small $\lambda_2,\gamma$ for the bounded-prior, anchoring, and calibration regularizers), and the scheduling/selection/scaling constants (the anneal length, the presence-weight temperature $s_\pi$, the negative-sampling ratio, the class weight $\omega$, the EMA momentum, the positional amplitude $\eta$, the conflict threshold $\kappa$, and the calibration margin $\kappa_{\mathrm{cal}}$), each of which is theoretically benign and is not swept. The agreement coefficient $w_\alpha$ is *learned* (initialized at $0$), not tuned. This keeps the effective search space to the four interpretable knobs above; a trimmed configuration that surfaces exactly these and pins the rest is provided as a preset.

## 3.10 Inference

Given a conversation, the model (1) encodes it with the relation-aware encoder, using the preliminary head's detached predicted labels for the emotion-transition edges; (2) computes the auxiliary probabilities and the relevance $\pi_{ij}$, selecting the evidence set $\mathcal{T}$; (3) estimates the necessity evidence for pairs in $\mathcal{T}$, re-scoring $f$ for the cause term and reusing the two pre-computed modality re-encodings (one per modality), applying pair-local perturbation only on the cause side; (4) predicts the distilled reasoning evidence with the student, without any LLM call; and (5) fuses the evidence with the complementarity-aware gate of Section 3.8—reliability-scaling the distilled vector (preserving $\hat\rho_{ij}$ undistorted) and conditioning on confidence, reliability, and necessity agreement—and adds the bounded positional prior and the necessity-modulated agreement term to output $p_{ij}$. Pairs outside $\mathcal{T}$ are scored from the pair representation with zero evidence, so all pairs receive a prediction.

**Complexity.** The encoder runs once per conversation, and the modality evidence adds two further encodings shared across all of its pairs; the cause-necessity term requires only a lightweight re-scoring of $f$ for the $M$ gated pairs. The LLM is used solely offline, so inference cost is dominated by a constant number of encoder passes plus the per-pair classifier, keeping the method practical for long conversations.
