---
title: Manage Household Waste using Open Source Data
date: 2023-05-27
categories:
  ["Household Waste", "Open Source data", "Sustainability", "Deep Learning"]
description: "Waste management is a major issue in our modern society. The increasing population and consumption lead to more waste being generated."
slug: "managing-household-waste-with-open-source-data"
image: "https://cdn-images-1.medium.com/max/3840/1*hrneNyOValFBbEAQbAqvwg.png"
---


Waste management is a major issue in our modern society. The increasing population and consumption lead to more waste being generated. The traditional method of disposing of household waste through landfilling or incineration is not sustainable in the long term. Using open-source data and a deep learning approach, a mobile app can manage household waste in an innovative way.

The global average of municipal solid waste generated per capita is <a href="https://datatopics.worldbank.org/what-a-waste/trends_in_solid_waste_management.html" target="_blank" rel="noopener">0.74kg per day</a>, but in high-income countries like the United States, it can be as high as 2.1kg per day. Proper waste management is essential to reduce the negative impact of household waste on the environment and human health, given the significant amount of waste generated every day.

### **Mobile Apps and Open Source Data**

Accessibility, collaboration, transparency, flexibility, security, and innovation are essential factors for creating successful and effective mobile apps that can be provided by using open source. <a href="https://openlittermap.com/" target="_blank" rel="noopener">OpenLitterMap</a> and <a href="https://www.trashout.ngo/" target="_blank" rel="noopener">TrashOut</a> apps are utilized for waste management and have made their source code accessible to the public.

<a href="https://deepwaste.my.canva.site/" target="_blank" rel="noopener">Deep Waste</a> is an open source mobile application that uses deep learning algorithms to classify and sort household waste. The app’s primary function is to enable users to take a photo of their waste and provide them with information on how to dispose of it properly. Various publicly available data sets such as <a href="https://github.com/garythung/trashnet" target="_blank" rel="noopener">TrashNet</a>, <a href="https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification" target="_blank" rel="noopener">Garbage Classification Data</a>, and <a href="https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2" target="_blank" rel="noopener">Garbage Classification V2</a> can be used in waste classification.


### **Inner Mechanics of Deep Waste**

In Deep Waste, they use the <a href="https://github.com/garythung/trashnet" target="_blank" rel="noopener">TrashNet</a> dataset and measure its performance with various state-of-the-art machine learning models: <a href="https://iopscience.iop.org/article/10.1088/1742-6596/1487/1/012008" target="_blank" rel="noopener">InceptionV3</a>, <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282336" target="_blank" rel="noopener">MobileNetV2</a>, <a href="https://ieeexplore.ieee.org/document/9563917" target="_blank" rel="noopener">InceptionResnet V2</a>, <a href="https://ieeexplore.ieee.org/document/10034869" target="_blank" rel="noopener">ResNet</a>, <a href="https://ieeexplore.ieee.org/document/9699161" target="_blank" rel="noopener">MobileNet</a>, and <a href="https://ieeexplore.ieee.org/document/9299017" target="_blank" rel="noopener">Xception</a> to learn and predict the type of waste.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/3840/1*hrneNyOValFBbEAQbAqvwg.png" alt="Deep Waste App, Suman Kunwar, CC BY-SA 4.0">
  <em>Deep Waste App, Suman Kunwar, CC BY-SA 4.0</em>
</p>


The app utilizes deep learning algorithms to recognize various types of waste, including plastics, glass, paper, and organic waste, based on the provided training data. Additionally, it can provide information on where to dispose of the waste and what recycling options are available. This approach can be customized to the user’s specific needs, including local waste management regulations and individual household waste disposal habits and preferences.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/4500/1*g9K5sMNKaflsTp5EhYzzyQ.png" alt="Deep Waste App Workflow, Suman Kunwar, CC BY-SA 4.0">
  <em>Deep Waste App Workflow, Suman Kunwar, CC BY-SA 4.0</em>
</p>


The classification models are then converted into a lite format, such as <a href="https://www.tensorflow.org/lite/guide" target="_blank" rel="noopener">TFLite</a>, which allows them to be used on mobile devices with limited resources. The lite format enables fast loading times, smaller size, and compatibility with various programming languages and platforms.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/21412/1*orzV49TVb-rPEc9MwEzjjw.png" alt="Waste Classification Results, Suman Kunwar, CC BY-SA 4.0">
  <em>Waste Classification Results, Suman Kunwar, CC BY-SA 4.0</em>
</p>

The app’s user interface is designed to be user-friendly and intuitive, making it easy for anyone to use, and can also be used in conjunction with other waste management services, such as waste collection and recycling services. Leaderboard is also generated <a href="https://www.stopwaste.co/calculator/" target="_blank" rel="noopener">based on the CO2</a> preserved by recycling/composting. The app can help households reduce the amount of waste they generate, increase their recycling rate, and reduce the amount of waste that ends up in landfills.

### **Problems in waste classification via image**

Classifying waste based on its image is challenging due to the varying composition of materials that can impact its recyclability or compostability, as well as the shapes and sizes of waste and the capabilities of local recycling centers. To ensure effective classification, it is necessary to establish a monitoring and improvement mechanism, such as feedback collection, additional data training, and leveraging advanced tools and technologies. So, continuous learning from new data and use of the state of art tools and technology makes the app more accurate and efficient over time. This ensures that the app can adapt to changes in waste management regulations and changes in household waste composition.

### **Conclusion**

In conclusion, the deep learning approach using open-source data to manage household waste through a mobile app is a promising solution to the growing waste management problem. It can help households reduce their waste generation, increase their recycling rate, and ultimately reduce the amount of waste that ends up in landfills. The app’s user-friendly interface and ability to adapt to changes make it a powerful tool in the fight against waste.

### Resources:

* iOS: <a href="https://apps.apple.com/app/deep-waste-ai/id6445863514?platform=iphone" target="_blank" rel="noopener">https://apps.apple.com/app/deep-waste-ai/id6445863514?platform=iphone</a>

* Android: <a href="https://play.google.com/store/apps/details?id=com.hai.deep_waste" target="_blank" rel="noopener">https://play.google.com/store/apps/details?id=com.hai.deep_waste</a>

* Source code: <a href="https://github.com/sumn2u/deep-waste-app" target="_blank" rel="noopener">https://github.com/sumn2u/deep-waste-app</a>
* Paper Link: <a href="https://www.aimspress.com/article/doi/10.3934/ctr.2023008?viewType=HTML" target="_blank" rel="noopener">https://www.aimspress.com/article/doi/10.3934/ctr.2023008</a>
