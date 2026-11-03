# Situation report: replay of a recorded event

> Replay of recorded data through the live service. It is an illustration, not a validation and not a forecast.

- Event: date-2020-12-01 (scene date 2020-12-01)
- Mean rainfall_3day over the event's 95 sampled rows: 60.0
- Wards shown: one recorded sample point per ward, at the median rainfall_3day of that ward's rows in the event
- Rows used: the primary frame of the level validation (2762 of 3062 mappable Block A rows, at locations not in the mappable training frame)
- Providers configured on the service: gemini, groq
- Generated (UTC): 2026-11-02T20:47:46Z

| Ward | rainfall_3day | Score | Level | Recorded label at the point |
|---|---|---|---|---|
| Ahero | 65.88970565795898 | 0.9778 | HIGH | GFM flooded |
| Awasi/Onjiko | 51.26750183105469 | 0.4151 | MEDIUM | GFM not flooded |
| East Kano/Wawidhi | 50.93618154525757 | 0.0099 | LOW | GFM not flooded |
| Kabonyo/Kanyagwal | 65.88970565795898 | 0.9534 | HIGH | GFM flooded |
| Kobura | 46.58156633377075 | 0.4316 | MEDIUM | GFM not flooded |

## Advisory by ward

### Ahero

- summary: This assessment for Ahero ward considers land cover, elevation, and rainfall over the previous three days.
- drivers:
  - The land cover is permanent water, which raises the score compared with the most common class in the training data.
  - The elevation here is below the typical training location, which raises the score.
  - The rainfall over the previous three days here is above the typical training location, which raises the score.
- meaning: The result ranks locations based on these factors and is not a warning.
- limits: This is a ranking score, not a flood probability: it comes from a model trained on a case-control sample, gives no lead time and is not an official warning. Follow county and national disaster authorities for guidance.

Source: gemini (gemini-3.1-flash-lite)
Providers tried: [{"provider": "gemini", "status": "accepted"}]

### Awasi/Onjiko

- summary: This assessment for Awasi/Onjiko considers elevation, rainfall over the previous three days, and soil clay content.
- drivers:
  - The elevation here is above the typical training location, which lowers the score.
  - The rainfall over the previous three days here is above the typical training location, which raises the score.
  - The soil clay content here is above the typical training location, which raises the score.
- meaning: The result ranks locations based on these factors and is not a warning. It provides a comparison of conditions across the area.
- limits: This is a ranking score, not a flood probability: it comes from a model trained on a case-control sample, gives no lead time and is not an official warning. Follow county and national disaster authorities for guidance.

Source: gemini (gemini-3.1-flash-lite)
Providers tried: [{"provider": "gemini", "status": "accepted"}]

### East Kano/Wawidhi

- summary: For East Kano/Wawidhi: this location's score is in the lowest third of the reference scores.
- drivers:
  - The distance to mapped surface water here is above the typical training location, which lowers the score.
  - The elevation here is above the typical training location, which lowers the score.
  - The rainfall over the previous three days here is above the typical training location, which raises the score.
- meaning: In the held-out validation, locations in higher bands were flooded more often than locations in lower bands. This ranks locations; it does not forecast an event.
- limits: This is a ranking score, not a flood probability: it comes from a model trained on a case-control sample, gives no lead time and is not an official warning. Follow county and national disaster authorities for guidance.

Source: template
Providers tried: [{"provider": "gemini", "reasons": ["timing"], "status": "rejected"}, {"provider": "groq", "reasons": ["timing"], "status": "rejected"}]

### Kabonyo/Kanyagwal

- summary: This assessment considers land cover, rainfall over the previous three days, and soil clay content for the Kabonyo/Kanyagwal ward.
- drivers:
  - The land cover is permanent water, which raises the score compared with the most common class in the training data.
  - The rainfall over the previous three days here is above the typical training location, which raises the score.
  - The soil clay content here is below the typical training location, which lowers the score.
- meaning: The result ranks locations based on these factors and is not a warning. It provides a comparison of conditions across different areas.
- limits: This is a ranking score, not a flood probability: it comes from a model trained on a case-control sample, gives no lead time and is not an official warning. Follow county and national disaster authorities for guidance.

Source: gemini (gemini-3.1-flash-lite)
Providers tried: [{"provider": "gemini", "status": "accepted"}]

### Kobura

- summary: This assessment for Kobura ward considers elevation, rainfall over the previous three days, and distance to mapped surface water.
- drivers:
  - The elevation here is above the typical training location, which lowers the score.
  - The rainfall over the previous three days here is above the typical training location, which raises the score.
  - The distance to mapped surface water here is above the typical training location, which lowers the score.
- meaning: This result ranks locations based on these factors and is not a warning. It provides information about the physical characteristics of the area.
- limits: This is a ranking score, not a flood probability: it comes from a model trained on a case-control sample, gives no lead time and is not an official warning. Follow county and national disaster authorities for guidance.

Source: gemini (gemini-3.1-flash-lite)
Providers tried: [{"provider": "gemini", "status": "accepted"}]

## Notices (verbatim from the service)

- Score meaning: ranking score from a case-control model; not a flood probability
- Level basis: third of the reference score distribution, validated in docs/PHASE_E_RESULTS.json
- Notice: Generated from a ranking score; not an official warning.
- Claim limit: scores rank locations inside GFM-mappable areas and are not flood probabilities; the sample is case-control
- Model as reported: {"arm": "hgb:con", "file": "nyando_hgbcon_5ae81ad8b030.onnx", "loaded": true, "protocol": "docs/PROMOTION_PROTOCOL.md (D20 promotion)", "registration_run": "fc48f612a09d4667a55fc06c3323bba8", "sha256": "5ae81ad8b0303c8a8bef79f177199e34f1d3cad87cb609afc6825a4b9980b723", "trained_on": "0b9283540d91154c0dda55b0d92cae7aa3cae6aeffad0390bbf77e028dd2063a"}

