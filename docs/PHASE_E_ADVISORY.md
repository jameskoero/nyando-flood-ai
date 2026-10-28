# Phase E advisory: contract and output guard (E2 and E3)

Register row D49. This document records what the advisory is, what it reads, what it may say, and what it never says. The protocol (`docs/PHASE_E_PROTOCOL.md` Section 5) fixes the provider order, the time limits and the cache; this document fixes the pure parts that the tests check.

## 1. What the advisory is

A plain-language reading of one location's ranking score from the registered model, with the score band validated in E1 (`docs/PHASE_E_RESULTS.json`) and the three features that move the score most. It is not a flood probability, gives no lead time and is not an official warning. Every advisory carries a fixed limits text added by code, whoever wrote the rest.

## 2. What it reads

A categorical context: an optional ward name, an optional band (LOW, MEDIUM or HIGH) and up to three drivers. A driver names a feature, whether it raises or lowers the score, and whether the value is above or below the typical training location (or, for land cover, which class). The context holds no number, so the prompt, the cache key and the template never see one.

## 3. Attribution

Attribution is by reference substitution: for each of the six features, the change in the score when that feature alone is replaced by its reference value (the training median, and the training mode for land cover). The three largest absolute changes are the drivers. It is computed from the same model that scores, through an injected scoring function. It is not SHAP.

## 4. The output guard

Text written by a provider is accepted only if its keys are exactly summary, drivers and meaning, with at most three drivers and the length limits in `src/advisory/guard.py`, and none of these rules fires:

- `digit`: any digit.
- `percent`: a percent sign or the word percent.
- `probability_word`: probability, chance, odds, likelihood and similar words.
- `action_instruction`: evacuation or any instruction to move or shelter.
- `certainty`: will flood, certain, guaranteed and similar words.
- `timing`: tomorrow, tonight, within, soon, hours, lead time and similar words.
- `severity`: extreme, severe, critical, dangerous, emergency, alert and similar words.
- `level_word`: the band names in capitals, or a risk level in words.
- `markup`: angle brackets, backticks, links.
- `injection`: text that tries to override the instructions.

A rejected text is replaced by the deterministic template, which is tested to pass the same guard for every combination it can be given.

## 5. What it does not claim

Land cover is the 2020 product and whether it leaks label information is not excluded (register row D14), so the advisory calls it a driver of the score, never a cause. No probability, no timing, no instruction to act, no comparison between wards and no statement about any event. The band reports how locations like this one were ranked in the held-out validation, a case-control sample, and nothing more.

## 6. Providers and fallback

Gemini first, then Groq, then the deterministic template (protocol Section 5). Each provider call is limited to 8 s and the whole request to 12 s; a provider asked after the time is spent is skipped. A provider is used only when both its settings exist: GEMINI_API_KEY and GEMINI_MODEL, GROQ_API_KEY and GROQ_MODEL. Model names are settings because providers retire models; no key or model name is stored in the repository.

A key is sent only in a request header. An error carries only its kind (timeout, http, network or shape) and an HTTP status, never a key, a URL or a response body. Provider text goes through the output guard before it is used.

An accepted provider answer is cached in memory for one hour by the hash of the prompt version and the categorical context. A template fallback is not cached, so the next request tries the providers again and an outage ends as soon as the providers answer.

## 7. Reference values

The reference for the attribution is the median of each numeric feature and the most common land-cover class (the smallest class code on a tie) of the mappable training frame: the flood rows and the controls outside the exclusion mask, the frame the E1 cutpoints were built on. The values are stored in docs/PHASE_E_REFERENCE.json with the training file's SHA-256, and are built from the real file by scripts/build_advisory_reference.py, which uses the standard library only.

tests/test_advisory_reference.py rebuilds them in CI and computes them a second way with pandas and the repository's own restrict_to_mappable; a difference fails the build. The builder writes nothing if the file's hash, its counts (4,420 rows, 1,970 floods, 3,475 mappable), its columns or its alignment with the layer-flag file differ from the verified ones.
