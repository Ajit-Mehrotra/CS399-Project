# CS399 Team 5 Group Project

## File Structure

```
.
├── README.md
├── scripts
│   ├── main.py
|   ├── fetch.py
|   ├── util.py
│   ├── preprocessing
│   │   ├── data_management.py
│   │   ├── split.py
│   │   ├── transformAndScale.py
│   │   ├── NLP.py
│   │   └── clean.py
|   └── analysis
|       ├── explore.py
│       ├── finetuneAndEvaluate.py
│       ├── reevaluate.py
|       └── trainAndEvaluate.py
├── data
│   ├── glassdoor_reviews.csv (raw data)
│   ├── train.csv (refactored data, preprocessed training set)
│   └── test.csv (refactored data, processed testing set)
├── models
│   ├── dt.pkl (Decision Tree)
│   ├── gaussian.pkl (Gaussian Naive Bayes)
|   ├── knn.pkl (K-Nearest Neighbors)
|   ├── logit_reg.pkl (Logistic Regression)
|   └── rfc.pkl (Random Forest Classifier)
├── fileserver (ignore)
│   └── main.go 
