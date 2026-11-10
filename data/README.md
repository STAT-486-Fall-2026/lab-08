# Pokémon data and variable guide

This lab uses the instructor-supplied copy of Rounak Banik's [The Complete Pokémon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon). The local `pokemon.csv` contains 801 catalog entries, 41 columns, generations 1–7, and the source's legendary-status labels. It is a historical snapshot, not a current inventory of every Pokémon or a sample of individual battles. We retain the file's original spelling, values, and missing entries.

SHA-256 of the exact supplied bytes:

```text
d11af5a43aeb5a3661060f1eddc5c63ec4b4657ffe0a8018ba5aa6b260b4c97a
```

The notebook reads this checked-in file and verifies its checksum. No Kaggle account, API key, or data download is needed. Package installation requires internet access.

## Pokémon primer

Pokémon are fictional creatures that trainers collect and use in battles. In the main games, a Pokémon has a species, one or two elemental **types**, numerical battle statistics, and other game characteristics. Types include familiar categories such as Fire and Water and fictional categories such as Ghost and Fairy. A Pokémon's type influences which attacks it resists or is vulnerable to; it is not its individual level or a numerical strength rating.

A trainer can gain experience, raise a Pokémon's level, and sometimes evolve it into a different species. An evolution is generally a separate catalog entry. The rows here describe catalog entries, rather than individual trained Pokémon. The numerical battle columns are **base statistics**: underlying characteristics used to calculate an individual's actual statistics. A particular individual's level and training also affect its battle statistics.

**Legendary** is a designation attached to particular Pokémon in the game world. It is not defined by crossing a fixed attack or total-stat threshold. Use the CSV's `is_legendary` labels as given; the historical source's grouping need not match every modern distinction between Legendary and Mythical Pokémon. Students are predicting the supplied label, not deciding which creatures deserve the designation.

## How to read this guide

The possible modeling relevance below is a reason to investigate a variable, not a claim about its fitted importance. A game-relevant attribute can receive little importance in a particular model. Likewise, a predictive association does not establish a cause. **L** means the legendary task; **T** means the primary-type task.

### Battle statistics: six different aspects of strength

The six base statistics are dimensionless game values. A larger value generally indicates greater capacity in that particular dimension; no single statistic guarantees that a Pokémon will win a battle. Physical and special moves use different offensive and defensive statistics. See the [official battle guide](https://diamondpearl.pokemon.com/en-gb/trainersguide/fundamentals/battling/).

| Column | Meaning and why it matters | Lab role |
| --- | --- | --- |
| `hp` | Base hit points: contributes to how much damage a Pokémon can withstand before fainting. It describes endurance, rather than current health during a battle. | Predictor in L and T. |
| `attack` | Base physical attack: contributes to damage dealt with physical moves. It can help describe an offensive battle profile. | Predictor in L and T. |
| `defense` | Base physical defense: helps reduce damage received from physical moves. It describes a different defensive capacity from HP. | Predictor in L and T. |
| `sp_attack` | Base special attack: contributes to damage from special moves. It is distinct from physical attack, so two Pokémon with similar totals may specialize differently. | Predictor in L and T. |
| `sp_defense` | Base special defense: helps reduce damage received from special moves. A Pokémon can be strong against physical damage and relatively weak against special damage. | Predictor in L and T. |
| `speed` | Base speed: influences move order; faster Pokémon generally act first when other move-order rules are equal. It does not measure distance traveled or attack damage. | Predictor in L and T. |
| `base_total` | Sum of the six base statistics above. It summarizes overall statistical capacity but hides how that capacity is distributed. | Audit its exact redundancy, then exclude it while retaining its six components. |

### Size, growth, and other game mechanics

These columns can describe physical appearance or game-design patterns. In particular, the lab tests whether apparent legendary prediction depends heavily on a few mechanics. A strong association is not automatically inappropriate: its usefulness depends on what information the intended prediction setting makes available.

| Column | Meaning and why it matters | Lab role |
| --- | --- | --- |
| `height_m` | Listed height in meters. It describes size, not level or evolutionary stage. Missing values are unknown measurements. Size may be associated with battle profiles or types; students must evaluate that possibility. | Predictor in L and T. |
| `weight_kg` | Listed mass in kilograms. Some game mechanics depend on weight, and body size may distinguish groups. Do not assume heavier always means stronger. | Predictor in L and T. |
| `percentage_male` | Percentage of a species reported as male, on a 0–100 scale. This is not an individual's sex. Missingness includes entries for which a male/female ratio is not applicable; zero and missing are not interchangeable. | Numeric predictor in L and T; preserve NaNs. |
| `base_egg_steps` | The source's hatch-time parameter, expressed as a number of walking steps. It is a catalog value, not an observed time-to-hatch experiment or a guarantee that every listed Pokémon can normally be bred. Game-specific hatch timing varies. | Predictor in T and full L; remove in restricted L. |
| `base_happiness` | Starting friendship value associated with the species when obtained under ordinary game rules. It is a game mechanic, not measured human happiness or a trainer's current relationship with an individual. | Predictor in T and full L; remove in restricted L. |
| `capture_rate` | Base catchability parameter, usually an integer on a scale up to 255. Higher values generally make catching easier. It is **not a percentage or a capture probability**; actual catching also depends on circumstances. Minior's entry contains two form-specific values, so numeric conversion makes that one entry NaN. | Predictor in T and full L; remove in restricted L. |
| `experience_growth` | Total experience points needed to reach level 100 under the species' growth curve. It is not the current experience or current level of an individual. It can distinguish growth categories. | Predictor in L and T. |
| `generation` | Number of the main game generation in which the Pokémon first appeared, here 1–7. It describes release history, not a Pokémon's age or strength. | Treat as a categorical predictor in L and T. |

The [PokéAPI species documentation](https://pokeapi.co/docs/v2#pokemon-species) provides additional explanations of catchability, starting friendship, sex ratios, hatch counters, and growth categories. It is supporting background; the lab does not replace the historical CSV with API values.

### Types, labels, and identifiers

| Column | Meaning and why it matters | Lab role |
| --- | --- | --- |
| `type1` | Primary elemental type, such as `grass` or `fire`. It is a nominal category, not an ordered score. Primary and secondary are recorded positions, not a claim that one type contributes a larger percentage. | Predictor in L; target in T. |
| `type2` | Optional secondary elemental type. A blank means the entry has only one recorded type; represent that legitimate absence as `no_secondary_type` when used as a predictor. | Predictor in L; excluded in T so another type label cannot help predict the target. |
| `is_legendary` | Source's binary designation: 1 for legendary and 0 otherwise. Rarity in this CSV concerns the class count; this column does not measure how often a player encounters a Pokémon. | Target in L; excluded in T. |
| `name` | English name identifying the catalog entry. It is useful for discussing mistakes, but feeding unique names to these models would undermine the intended attribute-based analysis. | Exclude from predictors; retain for error inspection. |
| `japanese_name` | Japanese name, sometimes with a romanized rendering. It identifies the entry rather than measuring a battle characteristic. | Exclude from predictors. |
| `pokedex_number` | Identifier in the National Pokédex, the game's catalog. Larger identifiers do not mean more powerful Pokémon; numbering can also encode release and family patterns. | Exclude from predictors. |
| `classfication` | Short descriptive species/category phrase, such as “Seed Pokémon.” The CSV really spells the column `classfication`. It is descriptive text, not the supervised-learning class label. | Exclude: high-cardinality text is outside this lab's scope. |
| `abilities` | Text representation of a list of possible passive traits, which can affect battles. Abilities are different from moves a trainer selects. The list is not a single clean categorical attribute or a record of which ability an individual currently uses. | Exclude: parsing lists and modeling abilities is outside this lab's scope. |

### All 18 defensive matchup columns

Each `against_*` value is the stored **incoming damage multiplier** for attacks of that column's type. A multiplier of 1 is neutral, 2 or 4 indicates increased vulnerability, 0.5 or 0.25 indicates resistance, and 0 indicates immunity in the recorded matchup. These are relative multipliers, not actual damage points or probabilities. For example, `against_fire = 2` says an incoming Fire attack is amplified relative to a neutral matchup; it does not say the Pokémon's own Fire attacks are stronger.

| Column | Incoming attack type |
| --- | --- |
| `against_bug` | Bug |
| `against_dark` | Dark |
| `against_dragon` | Dragon |
| `against_electric` | Electric |
| `against_fairy` | Fairy |
| `against_fight` | Fighting; note the shortened column suffix. |
| `against_fire` | Fire |
| `against_flying` | Flying |
| `against_ghost` | Ghost |
| `against_grass` | Grass |
| `against_ground` | Ground |
| `against_ice` | Ice |
| `against_normal` | Normal |
| `against_poison` | Poison |
| `against_psychic` | Psychic |
| `against_rock` | Rock |
| `against_steel` | Steel |
| `against_water` | Water |

These variables describe a defensive profile closely related to a Pokémon's types and potentially its other mechanics. Retain them in L. **Exclude every one in T**: the question is whether primary type can be predicted without this type-related matchup information. See [PokéAPI's type relationships](https://pokeapi.co/docs/v2#types) for background.

## Known snapshot details

- There are 70 legendary and 731 non-legendary entries; the source labels are retained without reclassification.
- Missing entries: `height_m` 20, `weight_kg` 20, `percentage_male` 98, and `type2` 384. The last category represents no secondary type rather than an unknown type.
- Minior has `capture_rate = "30 (Meteorite)255 (Core)"`. The raw CSV is preserved; the supplied feature helper coerces this entry to numeric NaN.
- `base_total` exactly equals the sum of the six base statistics for every row in this copy.
- Only Tornadus, Noibat, and Noivern have primary type `flying`. The supplied split allocates one to each partition. Per-class scores with one observation are particularly fragile.
- Forms and game rules can change across releases. Interpret the values as this supplied snapshot rather than silently updating selected entries from another source.
