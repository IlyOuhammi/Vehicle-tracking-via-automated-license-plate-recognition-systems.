# Vehicle-tracking-via-automated-license-plate-recognition-systems.
Notre projet consiste à suivre les véhicules en détectant les plaques d'immatriculation grâce à un filtre de Kalman, avec les étapes suivantes :
- Yolo pour détecter la plaque dans l'image.
- OCR pour lire l'immatriculation.
- Filtre de Kalman pour assurer un suivi précis du véhicule.

Bien que nous rencontrions des difficultés pour détecter certaines lettres arabes dans les immatriculations, cela ne gêne pas le fonctionnement du filtre de Kalman, car les chiffres sont les identifiants principaux. Même lorsque le modèle parvient à détecter les lettres arabes, ses performances sont limitées, surtout pour les vidéos. Nous avons utilisé :
- Un modèle pré-entraîné Paddle.
- Un OCR personnalisé (annotation des données et remplacement des lettres arabes par leurs équivalents latins).
- OCR avec Tesseract.
- TR-OCR, une solution basée sur des modèles Transformers pour la reconnaissance optique de caractères.
Le modèle a été déployé via une API .
ces fichiers python récapitulent notre travail:

🔹1- Car_detect.py:
Ce code détecte et suit les véhicules dans une vidéo en extrayant les plaques d'immatriculation via un modèle YOLO et en utilisant OCR pour lire le texte. Il applique ensuite un filtre de Kalman pour suivre la position du véhicule au fil des frames et génère une vidéo annotée avec les informations détectées.

🔹2- ocr.py:
Ce code lit un fichier Excel ("train.xlsx"), extrait les noms d'images à partir des chemins de fichiers dans la colonne "Image_Path", puis crée une nouvelle colonne "name" contenant uniquement le nom de chaque image (sans son extension). Ensuite, il sauvegarde le DataFrame modifié dans un nouveau fichier Excel ("train_set.xlsx").


🔹3- KalmanFilter.py:
Le code utilise le filtre de Kalman pour prédire la position d'objets détectés (ici, des véhicules) dans une vidéo. Il intègre également YOLO pour détecter les véhicules et utiliser leurs coordonnées pour appliquer le filtre de Kalman. Une version simulée génère des boîtes de détection pour tester sans YOLO, ajoutant un bruit aléatoire aux coordonnées. Les positions réelles et prédites sont affichées avec des couleurs distinctes pour visualiser l'efficacité du suivi.

🔹4- EasyOCR.ipynb:
Le script effectue la reconnaissance optique de caractères (OCR) à partir d'images. Il commence par générer des fichiers de boîte avec Tesseract pour extraire les coordonnées des caractères, puis utilise EasyOCR pour effectuer la lecture de texte dans une image. Après avoir appliqué des techniques de traitement d'image (comme le flou bilatéral et la détection de contours), il isole une région d'intérêt (ROI) contenant du texte, qu'il reconnaît et affiche sur l'image. Enfin, il affiche le texte extrait avec des annotations et un rectangle autour de la zone d'intérêt.

🔹5- Reconnaissance des caractères en utilisant tesseract.ipynb:
Ce code utilise Pytesseract pour la reconnaissance de texte dans des images, notamment des plaques d'immatriculation. Il commence par prétraiter l'image (conversion en niveaux de gris, lissage et binarisation). Ensuite, il détecte les plaques avec YOLO, puis extrait les caractères grâce à Pytesseract, supportant à la fois l'arabe et l'anglais(les chiffres).

🔹6- Moroccan_TrOCR.ipynb:
Ce notebook est conçu pour fine-tuner un modèle de reconnaissance de texte (OCR) basé sur TrOCR de Microsoft pour traiter des images en texte pour des données spécifiques. Il inclut l'importation des dépendances, le prétraitement des données, la création d'un dataset personnalisé, ainsi que l'entraînement et l'évaluation du modèle à l'aide de calculs de perte et du taux d'erreur de caractère (CER). Il permet de charger des ensembles de données OCR, d'effectuer un ajustement des hyperparamètres et de sauvegarder le modèle finement ajusté après chaque époque.

🔹7- finetuning_yolov10.ipynb: 
Ce code permet de télécharger les poids du modèle YOLOv10 et d'entraîner un modèle de détection d'objets. Il commence par cloner le dépôt YOLOv10 et installer les dépendances. Ensuite, il télécharge plusieurs fichiers de poids pré-entraînés. Après cela, il organise un jeu de données d'images et d'annotations en sous-dossiers pour l'entraînement, la validation et le test. Enfin, il configure un fichier YAML et lance l'entraînement du modèle YOLOv10 sur ces données.

🔹9- app2.py:
ce code utilise Streamlit pour créer une interface graphique. Lorsque l’utilisateur télécharge une vidéo dans l’interface, YOLO détecte la plaque, l’OCR extrait son contenu et le filtre de Kalman suit le véhicule. Les données (position, numéro de plaque) sont enregistrées instantanément et transformées par la suite en une vidéo de sortie que l’utilisateur peut télécharger. De plus, un fond personnalisé est ajouté à l'application à partir d'une image encodée en base64.
