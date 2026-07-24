# Session 06 Phase 1 -- Dataset Validation

## triviaqa

- N prompts: 9960  |  N beams: 99600

- Assert A (structure): PASS -- {'n_prompts': 9960, 'exactly_10_per_prompt': True, 'n_prompts_not_10': 0, 'zero_duplicate_prompts': True, 'n_duplicate_prompts': 0, 'spot_check_n': 20, 'spot_check_n_matched': 20, 'spot_check_pass': True, 'pass': True}

- Assert B (window): PASS -- empty=0.00%, length stats={'mean': 7.666214859437751, 'std': 7.936616575132172, 'min': 2, 'max': 64, 'median': 5.0}

- Assert C (round-trip): PASS -- 100/100 exact

- Assert D (determinism, from generation step): PASS

- Assert E (labels): PASS -- hallucination rate=53.5% (in 5-95% band: True), mixed=7458, all-truthful=1049, all-hallucinated=1453

- Assert F (manifest): PASS


### Eyeball examples

- [truthful] beam 53481 (rougeL=0.0492, bleurt_max=0.5324): 'The Crimea forms part of Ukraine.'
- [truthful] beam 58776 (rougeL=0.8667, bleurt_max=0.8301): 'The violin'
- [truthful] beam 6074 (rougeL=0.3333, bleurt_max=0.9029): 'Martin Clunes.'
- [truthful] beam 36662 (rougeL=0.7048, bleurt_max=0.9191): 'Terence Rattigan.'
- [truthful] beam 71874 (rougeL=0.0564, bleurt_max=0.63): 'The answer is Nuuk.'
- [hallucinated] beam 59863 (rougeL=0.0, bleurt_max=0.1448): 'Josep Borrell'
- [hallucinated] beam 50058 (rougeL=0.1722, bleurt_max=0.4884): 'Genus and Species'
- [hallucinated] beam 95244 (rougeL=0.1473, bleurt_max=0.4029): 'William Wordsworth was buried in Grasmere churchyard in 1850.'
- [hallucinated] beam 37038 (rougeL=0.0, bleurt_max=0.1614): 'Pontiac'
- [hallucinated] beam 58801 (rougeL=0.2589, bleurt_max=0.4995): 'The ocean or deep water'

### Decoding config
```json
{
  "do_sample": true,
  "num_beams": 10,
  "temperature": 0.5,
  "top_p": 0.99,
  "top_k": 5,
  "max_new_tokens": 64,
  "global_seed": 0,
  "prompt_seed_scheme": "sha256(global_seed:prompt_id)",
  "chat_template": null,
  "prompt_template": "Q: {question}\nA:",
  "rouge_threshold": 0.7,
  "sen_sim_threshold": 0.5,
  "bleurt_model": "BLEURT-20",
  "transformers": "5.13.0",
  "torch": "2.13.0+cu126",
  "cuda": "12.6"
}
```


---

## nq_open

- N prompts: 3610  |  N beams: 36100

- Assert A (structure): PASS -- {'n_prompts': 3610, 'exactly_10_per_prompt': True, 'n_prompts_not_10': 0, 'zero_duplicate_prompts': True, 'n_duplicate_prompts': 0, 'spot_check_n': 20, 'spot_check_n_matched': 20, 'spot_check_pass': True, 'pass': True}

- Assert B (window): PASS -- empty=0.00%, length stats={'mean': 17.42775623268698, 'std': 9.979500579067748, 'min': 2, 'max': 64, 'median': 17.0}

- Assert C (round-trip): PASS -- 100/100 exact

- Assert D (determinism, from generation step): PASS

- Assert E (labels): PASS -- hallucination rate=89.4% (in 5-95% band: True), mixed=1128, all-truthful=39, all-hallucinated=2443

- Assert F (manifest): PASS


### Eyeball examples

- [truthful] beam 32338 (rougeL=0.8333, bleurt_max=0.9342): 'At birth'
- [truthful] beam 15612 (rougeL=0.0, bleurt_max=0.6072): 'The Himalayan mountain range.'
- [truthful] beam 29529 (rougeL=0.5, bleurt_max=1.0031): 'National Industrial Recovery Act'
- [truthful] beam 34287 (rougeL=1.0, bleurt_max=0.8544): 'Atlantic Ocean'
- [truthful] beam 16554 (rougeL=0.7803, bleurt_max=0.7051): 'The Speaker of the House of Representatives.'
- [hallucinated] beam 1481 (rougeL=0.3077, bleurt_max=0.3785): 'The Dewey Decimal System was created by Melvil Dewey in 1876.'
- [hallucinated] beam 9480 (rougeL=0.2667, bleurt_max=0.4621): 'The continent of the Americas was named after Amerigo Vespucci, an Italian explorer.'
- [hallucinated] beam 35397 (rougeL=0.0556, bleurt_max=0.2472): 'The surname "Waters" is of English and Irish origin.'
- [hallucinated] beam 18660 (rougeL=0.1818, bleurt_max=0.3498): 'The movie "The King and I" was made in 1956.'
- [hallucinated] beam 17729 (rougeL=0.0, bleurt_max=0.1258): 'Ryan Giggs'

### Decoding config
```json
{
  "do_sample": true,
  "num_beams": 10,
  "temperature": 0.5,
  "top_p": 0.99,
  "top_k": 5,
  "max_new_tokens": 64,
  "global_seed": 0,
  "prompt_seed_scheme": "sha256(global_seed:prompt_id)",
  "chat_template": null,
  "prompt_template": "Q: {question}\nA:",
  "rouge_threshold": 0.7,
  "sen_sim_threshold": 0.5,
  "bleurt_model": "BLEURT-20",
  "transformers": "5.13.0",
  "torch": "2.13.0+cu126",
  "cuda": "12.6"
}
```


---

## tydiqa_gp

- N prompts: 440  |  N beams: 4400

- Assert A (structure): PASS -- {'n_prompts': 440, 'exactly_10_per_prompt': True, 'n_prompts_not_10': 0, 'zero_duplicate_prompts': True, 'n_duplicate_prompts': 0, 'spot_check_n': 20, 'spot_check_n_matched': 20, 'spot_check_pass': True, 'pass': True}

- Assert B (window): PASS -- empty=0.00%, length stats={'mean': 11.132727272727273, 'std': 8.275490747226899, 'min': 2, 'max': 64, 'median': 9.0}

- Assert C (round-trip): PASS -- 99/100 exact

- Assert D (determinism, from generation step): PASS

- Assert E (labels): PASS -- hallucination rate=47.0% (in 5-95% band: True), mixed=292, all-truthful=79, all-hallucinated=69

- Assert F (manifest): PASS


### Eyeball examples

- [truthful] beam 3035 (rougeL=1.0, bleurt_max=0.9458): 'Solid, liquid, gas, and plasma.'
- [truthful] beam 3286 (rougeL=0.8571, bleurt_max=0.8394): 'Answer:\n15 November 1877.'
- [truthful] beam 295 (rougeL=0.8889, bleurt_max=0.8066): 'The radius and the ulna.'
- [truthful] beam 2022 (rougeL=0.8571, bleurt_max=0.7435): 'At Tower Hill, London.'
- [truthful] beam 3966 (rougeL=0.2105, bleurt_max=0.5771): '8,850km (5,500mi)\nB: 6,259km (3,889mi)\nC: 21,196km (13,171mi)\nD: 13,171mi'
- [hallucinated] beam 4216 (rougeL=0.0, bleurt_max=0.0649): 'Jamal Julani was beaten unconscious and was in a critical condition.'
- [hallucinated] beam 3458 (rougeL=0.3333, bleurt_max=0.2577): '1848 and rebuilt in 1922.'
- [hallucinated] beam 2631 (rougeL=0.0, bleurt_max=0.1382): 'There is no information in the passage about the founding of Major League Baseball.'
- [hallucinated] beam 4124 (rougeL=0.2857, bleurt_max=0.3418): 'Over 3.'
- [hallucinated] beam 3057 (rougeL=0.2222, bleurt_max=0.3603): 'The best answer is: 1597.'

### Decoding config
```json
{
  "do_sample": true,
  "num_beams": 10,
  "temperature": 0.5,
  "top_p": 0.99,
  "top_k": 5,
  "max_new_tokens": 64,
  "global_seed": 0,
  "prompt_seed_scheme": "sha256(global_seed:prompt_id)",
  "chat_template": null,
  "prompt_template": "Concisely answer the following question based on the information in the given passage:\nPassage:\n{context}\nQ: {question}\nA:",
  "rouge_threshold": 0.7,
  "sen_sim_threshold": 0.5,
  "bleurt_model": "BLEURT-20",
  "transformers": "5.13.0",
  "torch": "2.13.0+cu126",
  "cuda": "12.6"
}
```
