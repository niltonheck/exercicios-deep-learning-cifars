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

- Acurácia para CIFAR-100 c/ Classes (Fine): 37,23%
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

- Class: apple | Accuracy: 66.00%
- Class: aquarium_fish | Accuracy: 41.00%
- Class: baby | Accuracy: 23.00%
- Class: bear | Accuracy: 30.00%
- Class: beaver | Accuracy: 5.00%
- Class: bed | Accuracy: 34.00%
- Class: bee | Accuracy: 42.00%
- Class: beetle | Accuracy: 32.00%
- Class: bicycle | Accuracy: 38.00%
- Class: bottle | Accuracy: 45.00%
- Class: bowl | Accuracy: 21.00%
- Class: boy | Accuracy: 17.00%
- Class: bridge | Accuracy: 34.00%
- Class: bus | Accuracy: 27.00%
- Class: butterfly | Accuracy: 40.00%
- Class: camel | Accuracy: 28.00%
- Class: can | Accuracy: 32.00%
- Class: castle | Accuracy: 61.00%
- Class: caterpillar | Accuracy: 31.00%
- Class: cattle | Accuracy: 26.00%
- Class: chair | Accuracy: 62.00%
- Class: chimpanzee | Accuracy: 52.00%
- Class: clock | Accuracy: 25.00%
- Class: cloud | Accuracy: 49.00%
- Class: cockroach | Accuracy: 58.00%
- Class: couch | Accuracy: 18.00%
- Class: crab | Accuracy: 33.00%
- Class: crocodile | Accuracy: 30.00%
- Class: cup | Accuracy: 54.00%
- Class: dinosaur | Accuracy: 37.00%
- Class: dolphin | Accuracy: 36.00%
- Class: elephant | Accuracy: 40.00%
- Class: flatfish | Accuracy: 28.00%
- Class: forest | Accuracy: 55.00%
- Class: fox | Accuracy: 31.00%
- Class: girl | Accuracy: 16.00%
- Class: hamster | Accuracy: 44.00%
- Class: house | Accuracy: 38.00%
- Class: kangaroo | Accuracy: 25.00%
- Class: keyboard | Accuracy: 31.00%
- Class: lamp | Accuracy: 28.00%
- Class: lawn_mower | Accuracy: 53.00%
- Class: leopard | Accuracy: 28.00%
- Class: lion | Accuracy: 43.00%
- Class: lizard | Accuracy: 11.00%
- Class: lobster | Accuracy: 16.00%
- Class: man | Accuracy: 26.00%
- Class: maple_tree | Accuracy: 51.00%
- Class: motorcycle | Accuracy: 72.00%
- Class: mountain | Accuracy: 48.00%
- Class: mouse | Accuracy: 14.00%
- Class: mushroom | Accuracy: 41.00%
- Class: oak_tree | Accuracy: 63.00%
- Class: orange | Accuracy: 60.00%
- Class: orchid | Accuracy: 55.00%
- Class: otter | Accuracy: 8.00%
- Class: palm_tree | Accuracy: 56.00%
- Class: pear | Accuracy: 36.00%
- Class: pickup_truck | Accuracy: 36.00%
- Class: pine_tree | Accuracy: 31.00%
- Class: plain | Accuracy: 67.00%
- Class: plate | Accuracy: 45.00%
- Class: poppy | Accuracy: 44.00%
- Class: porcupine | Accuracy: 44.00%
- Class: possum | Accuracy: 13.00%
- Class: rabbit | Accuracy: 20.00%
- Class: raccoon | Accuracy: 23.00%
- Class: ray | Accuracy: 26.00%
- Class: road | Accuracy: 70.00%
- Class: rocket | Accuracy: 53.00%
- Class: rose | Accuracy: 40.00%
- Class: sea | Accuracy: 49.00%
- Class: seal | Accuracy: 5.00%
- Class: shark | Accuracy: 46.00%
- Class: shrew | Accuracy: 21.00%
- Class: skunk | Accuracy: 71.00%
- Class: skyscraper | Accuracy: 64.00%
- Class: snail | Accuracy: 24.00%
- Class: snake | Accuracy: 12.00%
- Class: spider | Accuracy: 34.00%
- Class: squirrel | Accuracy: 20.00%
- Class: streetcar | Accuracy: 31.00%
- Class: sunflower | Accuracy: 71.00%
- Class: sweet_pepper | Accuracy: 34.00%
- Class: table | Accuracy: 24.00%
- Class: tank | Accuracy: 55.00%
- Class: telephone | Accuracy: 50.00%
- Class: television | Accuracy: 58.00%
- Class: tiger | Accuracy: 16.00%
- Class: tractor | Accuracy: 43.00%
- Class: train | Accuracy: 22.00%
- Class: trout | Accuracy: 50.00%
- Class: tulip | Accuracy: 16.00%
- Class: turtle | Accuracy: 20.00%
- Class: wardrobe | Accuracy: 76.00%
- Class: whale | Accuracy: 33.00%
- Class: willow_tree | Accuracy: 45.00%
- Class: wolf | Accuracy: 42.00%
- Class: woman | Accuracy: 27.00%
- Class: worm | Accuracy: 8.00%