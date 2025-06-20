# ISAS2025
## Recognize the Unseen: Unusual Behavior Detection from Pose Data

Facilities that support individuals with developmental disabilities often struggle to monitor and identify unusual behaviors due to limited staffing and the subtle, irregular nature of such actions. Traditional observation-based methods are time-consuming, error-prone, and insufficient for real-time intervention. To address this gap, we introduce a new challenge focused on developing machine learning models that can recognize abnormal behaviors using only pose estimation data extracted from video.


This challenge deals with unusual behaviour recognition from activities mimicking that of developmentally disabled individuals. Pose keypoints are extracted from video recordings of healthy participants simulating eight real-world activities—four normal (eating, sitting, walking, using a phone) and four unusual (head banging, throwing objects, attacking others, and biting hands/fingers). These activities were designed in consultation with staff at a real care facility and reflect common behaviors that require monitoring and intervention.

## CHALLENGE GOAL

Classify normal vs. unusual activities using skeleton data of the test data.

Develop a general model across individuals using a Leave-One-Subject-Out (LOSO) evaluation.

This challenge offers a unique opportunity to work on a real-world problem that blends human behavior understanding, healthcare support, and pose-based noninvasive approach.

### EVALUATION GUIDELNES

Classification of activities

Accuracy, F1-Score (Abnormal), Precision and Recall (Abnormal)

Final score will prioritize F1-Score on abnormal class

Model performance on LOSO evaluation

After finishing the classification, we will provide the labels for 5 person evaluation with Leave One Subject Out