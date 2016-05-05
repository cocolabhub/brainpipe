# To do
## Documentation
- Physio
- cfc

## Modules
### Commun :
- Output comme array serait plus claire (features et classification) 

### bpstudy
- add_dataset
- add_file

### Features
- Cfc :
	- Changer la façon de gérer les signaux d'entrée car cfc ne calcul pas juste du couplage phase amplitude. inclure x1={'kind':'phase', 'cycle':3}, x2={'kind':'amplitude', 'cycle':6} et dans le get() et statget() virer la possibilité de calculer en local qui n'est qu'un cas particulier
	- Méthodes CFC :
		- Phase synchrony (WARNING : CFC en double phase)
		- Amplitude PSD (WARNING : CFC en double amplitude)
		- ndPAC
	- Méthodes normalisation :
		- Time lag
		- Swap phase/amplitude
		- Swap amplitude
		- Circular shifting	
- welch method for power
- Prefered phase (partage de fonctions avec pac // phase // amplitude)
- Amplitude simple (partage de fonctions avec power)
- Power phase locked
- Inclure pentropy, kurtosis... Du code de Tarek
- Connectivité (Philippe)

### Panda complements
- pdOrder? pdPLot?

### Physiology
- Bug : impossible de sauvegarder un pd2physio. Inclure les méthodes dans la classe physio?
- Plot brain structures?
- Order?

### Classification
- Ok

### Visualization
- global 2D plot for TF and CFC, temporal generalization
- Time generalisation

### Multifeatures
- Améliorer la parallélisation (sur rep en cas de non groupe)

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

