


Objectif:
	Reconnaissance des chiffres formés par des gestes de la main en utilisant les algorythmes d'apprentissage artificiel
	
Contexte: 
	Enfants dys utilisent ces stratégies pour améliorer leur apprentissage des nombres et du calcul
	
Stratégie:
	1- Collecte d'une base de donnée d'images annotées 
	2- Extraction des points repères de la main des images
	3- Normalisation et préparation des données issues des points repères
	4- Entrainement du réseau de neurone
	5- Tests et prédiction 
	
	
	
1- Collecte d'une base de donnée d'images annotées 
	Source de données HaGrid (kaggle.com), 500K images.
	18 classes correspondant à 18 gestes de la main identifiés.

2- Extraction des points repères de la main des images
	Option 1: Utilisation des libraries MediaPipe (traitement fonctionnel sur un échatillon restreint, mais nécessite trop de ressources pour 500K images).
	Option 2: Exploitation des annotations qui intègrent directement les points repères de la main.
	
3- Normalisation et préparation des données issues des points repères
	Toutes les coordonnées sont fournies en données relatives à la taille de l'image.
	Nous souhaitons traiter les distances entre les points repères et le point repère de référence (poignet).
	Les données étant toutes relatives à la taille de l'image, les distances calculées sont faibles et donc non significatives.
	Je souhaite transformer ces données en des distances relatives à la taille du rectangle encadrant la main. 
	Deux conséquences attendues: 
		- Les distances sont significativement différentes entre-elles
		- Les écarts de distances liés aux tailles de mains différentes sont gommés
	
	Exemple d'annotation:
	"0035fcf9-da92-4167-a914-8cf6ac753160":
    {
        "bboxes": // liste des rectangles autour des mains détectées
        [
            [
                0.48116749,		// Top left X position, relative image coordinate
                0.2571216,		// Top left Y position, relative image coordinate
                0.19981102,		// width, relative image coordinate
                0.10427412		// heigh, relative image coordinate
            ]
        ],
        "labels":
        [
            "fist" // signe - nombre identifié
        ],
        "landmarks": // liste des points repères de chaque mains détectée
        [
            [	
                [ 	// point repère 0
                    0.6117590754152344, // X position, relative image coordinate
                    0.34647556464205115	// Y position, relative image coordinate
                ],
				...
                [	// point repère 20 
                    0.6275607276425796,
                    0.3149215676785891
                ]
            ]
        ],
        "leading_conf": 1.0,
        "leading_hand": "left",
        "user_id": "d5bfbd8885ce2a0131a99e479420076f"
	}



	Xbox / Wimage
	Ybox / Himage
	Wbox / Wimage
	Hbox / Himage



	Xlan / Wimage
	Ylan / Himage





