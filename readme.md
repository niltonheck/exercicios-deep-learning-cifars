Códigos para os Exercícios de Deep Learning
====

Versão de Python utilizada: **3.6.7**


Arquivos:
----
  
- cnn_cifar100-(coarse|fine).py: Compila o modelo, exporta as melhores deifinições de peso, e apresenta o resultado geral de acurácia. Módulo baseado nos rótulos das superclasses/classes do dataset;

- cnn_cifar100-(coarse|fine)-test.py: Executa os modelos para cada superclasse ou classe, apresentando os resultados específicos para cada classe, além de gerar ao menos 2 imagens de erros de predição;

- my_utils.py: Métodos úteis para a atividade;

- weights-(coarse|fine).best-hdf5: Os melhores pesos encontrados para o modelo.

- requirements.txt: Lista das bibliotecas obrigatórias para execução das redes convolucionais.

Resultados Alcançados:
---

- Acurácia para CIFAR-100 c/ Classes (Fine): 39,50%
- Acurácia para CIFAR-100 c/ Superclasses (Coarse): 51,17%

**Acurácia por Superclasse**

- Class: aquatic mammals | Accuracy: 54.80%
- Class: fish | Accuracy: 46.80%
- Class: flowers | Accuracy: 75.20%
- Class: food containers | Accuracy: 50.20%
- Class: fruit and vegetables | Accuracy: 55.80%
- Class: household electrical devices | Accuracy: 41.60%
- Class: household furniture | Accuracy: 53.80%
- Class: insects | Accuracy: 54.40%
- Class: large carnivores | Accuracy: 44.40%
- Class: large man-made outdoor things | Accuracy: 59.80%
- Class: large natural outdoor scenes | Accuracy: 70.00%
- Class: large omnivores and herbivores | Accuracy: 38.20%
- Class: medium-sized mammals | Accuracy: 42.00%
- Class: non-insect invertebrates | Accuracy: 24.20%
- Class: people | Accuracy: 62.20%
- Class: reptiles | Accuracy: 27.40%
- Class: small mammals | Accuracy: 37.40%
- Class: trees | Accuracy: 78.60%
- Class: vehicles 1 | Accuracy: 61.00%
- Class: vehicles 2 | Accuracy: 47.40%

**Acurácia por Classe**

- Class: apple | Accuracy: 68.00%
- Class: aquarium_fish | Accuracy: 53.00%
- Class: baby | Accuracy: 28.00%
- Class: bear | Accuracy: 18.00%
- Class: beaver | Accuracy: 18.00%
- Class: bed | Accuracy: 37.00%
- Class: bee | Accuracy: 34.00%
- Class: beetle | Accuracy: 41.00%
- Class: bicycle | Accuracy: 52.00%
- Class: bottle | Accuracy: 48.00%
- Class: bowl | Accuracy: 28.00%
- Class: boy | Accuracy: 29.00%
- Class: bridge | Accuracy: 35.00%
- Class: bus | Accuracy: 24.00%
- Class: butterfly | Accuracy: 19.00%
- Class: camel | Accuracy: 36.00%
- Class: can | Accuracy: 43.00%
- Class: castle | Accuracy: 59.00%
- Class: caterpillar | Accuracy: 37.00%
- Class: cattle | Accuracy: 24.00%
- Class: chair | Accuracy: 61.00%
- Class: chimpanzee | Accuracy: 53.00%
- Class: clock | Accuracy: 35.00%
- Class: cloud | Accuracy: 55.00%
- Class: cockroach | Accuracy: 63.00%
- Class: couch | Accuracy: 29.00%
- Class: crab | Accuracy: 31.00%
- Class: crocodile | Accuracy: 40.00%
- Class: cup | Accuracy: 59.00%
- Class: dinosaur | Accuracy: 38.00%
- Class: dolphin | Accuracy: 49.00%
- Class: elephant | Accuracy: 43.00%
- Class: flatfish | Accuracy: 24.00%
- Class: forest | Accuracy: 49.00%
- Class: fox | Accuracy: 38.00%
- Class: girl | Accuracy: 19.00%
- Class: hamster | Accuracy: 41.00%
- Class: house | Accuracy: 31.00%
- Class: kangaroo | Accuracy: 30.00%
- Class: keyboard | Accuracy: 32.00%
- Class: lamp | Accuracy: 35.00%
- Class: lawn_mower | Accuracy: 59.00%
- Class: leopard | Accuracy: 37.00%
- Class: lion | Accuracy: 43.00%
- Class: lizard | Accuracy: 16.00%
- Class: lobster | Accuracy: 22.00%
- Class: man | Accuracy: 28.00%
- Class: maple_tree | Accuracy: 45.00%
- Class: motorcycle | Accuracy: 63.00%
- Class: mountain | Accuracy: 48.00%
- Class: mouse | Accuracy: 23.00%
- Class: mushroom | Accuracy: 24.00%
- Class: oak_tree | Accuracy: 70.00%
- Class: orange | Accuracy: 64.00%
- Class: orchid | Accuracy: 47.00%
- Class: otter | Accuracy: 14.00%
- Class: palm_tree | Accuracy: 47.00%
- Class: pear | Accuracy: 42.00%
- Class: pickup_truck | Accuracy: 44.00%
- Class: pine_tree | Accuracy: 30.00%
- Class: plain | Accuracy: 70.00%
- Class: plate | Accuracy: 61.00%
- Class: poppy | Accuracy: 52.00%
- Class: porcupine | Accuracy: 46.00%
- Class: possum | Accuracy: 20.00%
- Class: rabbit | Accuracy: 24.00%
- Class: raccoon | Accuracy: 29.00%
- Class: ray | Accuracy: 42.00%
- Class: road | Accuracy: 74.00%
- Class: rocket | Accuracy: 65.00%
- Class: rose | Accuracy: 46.00%
- Class: sea | Accuracy: 59.00%
- Class: seal | Accuracy: 17.00%
- Class: shark | Accuracy: 23.00%
- Class: shrew | Accuracy: 23.00%
- Class: skunk | Accuracy: 57.00%
- Class: skyscraper | Accuracy: 66.00%
- Class: snail | Accuracy: 20.00%
- Class: snake | Accuracy: 15.00%
- Class: spider | Accuracy: 39.00%
- Class: squirrel | Accuracy: 21.00%
- Class: streetcar | Accuracy: 45.00%
- Class: sunflower | Accuracy: 62.00%
- Class: sweet_pepper | Accuracy: 27.00%
- Class: table | Accuracy: 22.00%
- Class: tank | Accuracy: 58.00%
- Class: telephone | Accuracy: 42.00%
- Class: television | Accuracy: 45.00%
- Class: tiger | Accuracy: 25.00%
- Class: tractor | Accuracy: 49.00%
- Class: train | Accuracy: 31.00%
- Class: trout | Accuracy: 46.00%
- Class: tulip | Accuracy: 29.00%
- Class: turtle | Accuracy: 20.00%
- Class: wardrobe | Accuracy: 75.00%
- Class: whale | Accuracy: 35.00%
- Class: willow_tree | Accuracy: 38.00%
- Class: wolf | Accuracy: 40.00%
- Class: woman | Accuracy: 16.00%
- Class: worm | Accuracy: 24.00%