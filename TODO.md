# To do
## Documentation
- Physio

## Modules
### bpstudy
- add_dataset
- add_file

### Features
- add cfc and the diffrents methods
- welch method for power
- inherate from built-in classes to avoid one unecessary line for plotting (plot method should be integrate in the power object)
- prefered phase
- power phase locked
- Inclure pentropy, kurtosis... Du code de Tarek

### Panda complements
- pdOrder? pdPLot?

### Physiology
- Plot brain structures?
- Order?

### Classification
- Choix d'un classifieur
- Choix d'une cross-val
- Choix d'une évaluation statistique:
	- Binomiale
	- Permutations:
		- shuffle labels
		- Shuffle intra-class
		- Shuffle "full randomization"
- kind : sf ou mf

### Visualization
- global 2D plot for TF and CFC
- Time generalisation

### Multifeatures
- TO DO
- Ajouter la possibilité de classer des groupes (pas facile du tout de combiner avec le single)

## Organization
- Find a incremental template name for saving files
- Fichier tampon pandas Dataframe qui résumé les paramètres // path // 
- fonction seek pour charger que ce dont on a besoin

## Test
- Add a test_physio
- Add notebooks

# Future
- Parallelize some functions
- Brain3D
- Nipype integration
- Operation in cache
- cpu/gpu
- Neural networks/DeepLearning (Philippe)

