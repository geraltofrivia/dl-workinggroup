# Text arc — planning notes

Good — I've got the whole course in context, so let me reason about text from where your students actually *are*, not from where an NLP syllabus usually starts.

## The one idea your course already set up for you

Lesson 5.3 is your secret weapon. You taught that **the architecture has to match the structure of the data** — an MLP on pixels fails because it ignores spatial locality; convolution works because it exploits it. That's the exact lens for text. The opening question of the whole arc writes itself:

> "Images had spatial structure that convolution exploited. What structure does text have — and why can't we just do the pixel thing again?"

That reframes your instinct ("do W2V to talk about how representing words ≠ representing images") as not jumping the gun at all — it's the *correct* first principle. Representation is genuinely the first novel problem, and it's the one thing that doesn't reduce to what they already know.

So I'd anchor everything on two threads you can carry across every session:
1. **Representation** — how does a discrete symbol become a vector? (the new thing)
2. **"Architecture matches data structure"** — the 5.3 refrain, now applied to sequences instead of grids.

## My blunt opinions on your candidate paths

**Cut HMM.** It's historically real but it's a different paradigm — probabilistic, EM, no gradient descent, no `nn.Module`. It doesn't connect to the spine you spent five lessons building, and for a zero-homework beginner group it's a full session spent on a dead end. The only thing it buys you ("sequences have structure you can model") you get for free later.

**Don't do the full history as a tour.** `HMM → W2V → RNN → LSTM → cross-attn → self-attn → transformer` is ~7 sessions, and half of it exists only to motivate the next step. History is seductive but expensive. You can get the motivational payload of "people used recurrence and it struggled with long sentences" in *one sentence*, not one session. Compress ruthlessly.

**Treat RNN/LSTM as a foil, not a destination.** Spend one session on "how do you process a sequence at all," show the recurrent hidden-state idea and its bottleneck, and use that failure to *set up attention*. Don't dwell on LSTM gate math — "an LSTM is an RNN with learnable valves that decide what to remember and forget" is enough intuition. Gating math is a rabbit hole with low payoff for this audience.

**Self-attention before cross-attention.** Historically cross-attention (translation alignment) came first, but for your group "every word looks at every other word" is cleaner than the whole seq2seq encoder-decoder setup. Make cross-attention a footnote if at all.

## The spine I'd actually run

This is the fail-first arc, exactly in your house style:

| # | Session | Fail-first hook → payoff |
|---|---|---|
| **T1** | **Representing words** | One-hot: sparse, huge, and `cat·dog == cat·airplane == 0` — *zero* similarity structure. **Fail.** → dense learned embeddings. Punchline that ties to their world: **an embedding is just a linear layer applied to a one-hot vector** — a learnable lookup table. Nothing new, just weights. |
| **T2** | **Where meaning comes from (W2V)** | Random embeddings are meaningless. How do they get meaning? "You know a word by the company it keeps." Train vectors to predict their neighbors → similar contexts pull vectors together. Payoff: nearest neighbours + `king − man + woman ≈ queen`. Big idea: **self-supervision** — meaning learned by gradient descent on a fake task. |
| **T3** | **Sequences (the bottleneck)** | Now average the word vectors → bag of words. But "dog bites man" == "man bites dog". **Fail — order lost.** → recurrence: a hidden state that reads left to right (memory). Then its own failure: one fixed vector can't hold a long sentence; distant words can't connect. |
| **T4** | **Attention (the payoff)** | Fix the bottleneck: let *every* position look directly at *every* other and decide what's relevant. **Bridge from 5.3:** convolution mixes local neighbours with *fixed* weights; attention mixes *all* positions with *content-dependent* weights. Query/Key/Value as a soft dictionary lookup. |
| **T5** (optional) | **Transformers** | Attention alone is order-blind — it's a *set* operation (nice mini fail-first!) → positional encoding. Add the FFN + stacking, and "this is what's inside an LLM." |

That's **4 sessions to attention, 5 to transformers.** HMM gone, LSTM as a one-session foil.

## The one decision that determines everything: where's the finish line?

Three honest stopping points, each a *complete* unit:

- **Stop at T2 (embeddings).** 2–3 sessions, deeply satisfying, honors "few concepts, go deep." You'd never reach attention, but representation alone is genuinely novel and rich. Lowest risk.
- **Stop at T4 (attention).** My recommended target. Reaches the idea that matters today, and everything you've built points at it.
- **Stop at T5 (transformers).** The full modern payoff, but see the warning below.

**One caveat I want to be straight about:** attention from scratch is the *hardest* thing this group will have done — QKV, a softmax over positions, and the tensor shapes are a real step up, in 1h with no homework. It's doable, but T4/T5 will be dense and you may need to lean on the pre-filled notebook more than usual. If the group's appetite is more "curious dabblers" than "hungry for the deep end," stopping at embeddings and doing attention as a later "part 2" is not a cop-out — it's the right read of the room.

## A note on your v1

Your deleted branch went `one-hot sentiment → POS tagging → W2V → RAG` — i.e. representation-first, then it *jumped straight to RAG* and treated the LLM as a black box, skipping the entire RNN→transformer story. So "textv2" is really a fork in the road: **keep it representation-pragmatic (and maybe end at RAG again), or actually open the black box this time and build up to attention.** Given how well 5.3 sets up attention, I think v2 is your chance to open the box.

---

So my actual opinion, in one line: **anchor on representation, cut HMM, use RNN as a foil, and aim for self-attention as the finish — but be honest with yourself about whether the group wants to climb that last hill or stop at embeddings.**

Which finish line feels right to you — embeddings, attention, or all the way to transformers? Once you pick, I'll help you break the target session down into the fail-first beats and figure out what's live-coded vs. pre-filled.
