# mind-over-motor
Logistic regression classifier that classifies target limb movement based on brain activity data collected through EEG.


# Group members:
Fiona Stern and Beckett O’Reilly

# Abstract:
Our project will address the disconnect between our current understanding of brain signals and their relationship to bodily movement, particularly in brain-computer interfaces to aid those with movement-limiting injuries (e.g., stroke, paralysis) in being able to once again use their own muscles. We will aim to classify the target limb movement based on brain activity data collected through EEG. First, we will engineer the EEG data into features that we will use to perform logistic regression to classify the target movements.  We will evaluate our multinomial classifier's success using accuracy, since the cost for the different prediction errors is not variable.

# Motivation and question:
We have EEG data for the brainwaves of people with brain damage of varying location and severity. The brainwaves measure the electrical activity while thinking about or actually moving their right and left arms. We would like to predict what type of movement (e.g., limb and direction) is based on these brainwaves. This type of prediction could be used in brain-computer interfaces that could help those who’ve experienced some type of limb limitation due to a brain injury in using that/those limb(s) again.

# Planned deliverables:
The deliverables will include a Python package containing the data pipeline and the model implementation. With full success, this package will contain everything required to extract the data and train and run the model, ensuring it consistently performs well on unseen data. With partial success, the Python package will contain these elements, but the model doesn’t necessarily perform well on unseen data.

The deliverables will also include a Jupyter notebook where we implement the model to analyze the dataset and apply our evaluation metrics to show the model's performance. With full success, the notebook will contain these elements with an in-depth analysis and proof of strong performance. With partial success, the notebook will display an implementation that might have less-than-ideal performance.

# Resources required:
To complete our project, we will need access to a dataset that contains information about which limb a person is moving and the accompanying EEG signals.  We have found this information in the following dataset: https://doi.org/10.6084/m9.figshare.21679035.  We anticipate that the computing power on our personal computers will be adequate to process the data.  To collaborate with each other, we intend to use GitHub, as specified in the project instructions.  At the moment, we do not foresee the need for additional resources.

# What You Will Learn:
In this project, we intend to learn a variety of different skill sets.  Fiona, having never worked on a project of this sort before, intends to learn the basics of collaborating on code, processing an external dataset, and working with EEG data.  She believes that through this project, she will learn to process time series data effectively, creating features that can be used in machine learning.  She is also expecting to learn how to deal with the unexpected as she gets immersed in the project.  Beckett also intends to become familiar with the intersection of machine learning and brain signals.  Having previously only worked with EEGs in a psychology laboratory setting, he intends to learn how to create an algorithm to process the data effectively.  He also intends to improve his overall teamwork, project management, and alarm clock waking abilities. We both intend to learn the practical applications of the algorithms learned in this class, particularly in understanding what sort of algorithm might be most effective to accomplish our goals.

# Risk statement:
A successful model may require many parameters, and we run the risk of overfitting our training data when working with a relatively small dataset. This risk is particularly relevant if we choose to switch to a neural network.

It can be tricky to use EEG outputs as data that can be fed into a machine. Another risk we might run into is creating a successful pipeline from which the signal can even be extracted in any meaningful way.

# Ethics Statement:
If our project is successful and deployed, we would hope that it allows for BCI technology that will allow stroke patients to have greater control over their limbs.  The primary beneficiaries of this technology would be individuals who are recovering from a stroke who do not have full control over both sides of their bodies. Those who do not have access to high-quality healthcare or expensive technologies might be excluded from these benefits.  We do not expect that anyone would be harmed by our project.  Overall, we expect that the world would become a better place if our project is successful.  This conclusion rests on the assumptions that stroke patients’ lives would improve with a greater mind-body connection and that the world is a better place when stroke patients’ lives are improved.
