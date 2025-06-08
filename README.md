


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
                0.48116749,		// Top left X position, relative image coordinate, vXib0 = Xb0/Wi
                0.2571216,		// Top left Y position, relative image coordinate, vYib0 = Yb0/Hi
                0.19981102,		// width, relative image coordinate, vWib0 = Wb01/Wi
                0.10427412		// heigh, relative image coordinate, vHib0 = Hb01/Hi
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
                    0.6117590754152344, // X position, relative image coordinate, vXil0 = Xl0/Wi
                    0.34647556464205115	// Y position, relative image coordinate, vYil0 = Yl0/Hi
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


	Valeurs recherchée vXbl0 = (Xl0 - Xb0)/Wb0 ('v'aleur de 'X' relative à la taille de la 'b'oxe et non de l'imagepoint repère, du 'l'andmark 0)
		Or Wb0 = vWib0 * Wi
		Donc vXbl0 = (Xl0 - Xb0)/(vWib0 * Wi) = (Xl0/Wi - Xb0/Wi) * 1/vWib0
        Or Xl01/Wi - Xb01/Wi =  vXil0 - vXib0

        D'où :
            vXbl0 = (vXil0 - vXib0)/vWib0

		
		 exemple : (0.6117590754152344 - 0.48116749) * (1/0.19981102) = 0,653575491

    On trouve donc :
        vYbl0 = (vYil0 - vYib0) / vHib0


# boucle pour entrainer le réseau sur ttes les données 'epochs' fois
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = np.array(x_train[j])
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # calcule l'erreur ( uniquement pour l'affichage )
                err += self.loss(y_train[j], output)

                # retroprogation du gradient de l'erreur
                error = self.loss_derivee(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calcule l'erreur moyenne sur tous les échantillons
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))