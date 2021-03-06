## Titanic with Random Forests

So, decision trees [are OK](./dt_analysis.md), but I think we can do better. Start with the same steps as before:

````julia
using DataFrames

train = readtable("data/train.csv")
tst = readtable("data/test.csv")

train[isna(train[:Age]), :Age] = mean(dropna(train[:Age]))
train[isna(train[:Fare]), :Fare] = mean(dropna(train[:Fare]))

countmap(train[:Embarked])
train[isna(train[:Embarked]), :Embarked] = "S"

using DecisionTree

features = Array(train[:, [:Age, :Fare, :Sex, :SibSp, :Pclass, :Parch]])
labels = Array(train[:Survived])

model = build_forest(labels, features, 3, 10)

tst[isna(tst[:Age]), :Age] = mean(dropna(tst[:Age]))
tst[isna(tst[:Fare]), :Fare] = mean(dropna(tst[:Fare]))

testarray = Array(tst[:, [:Age, :Fare, :Sex, :SibSp, :Pclass, :Parch]])

predictions = apply_forest(model, testarray)

df = DataFrame(PassengerId = tst[:PassengerId], Survived = predictions)
writetable("data/rf_predictions.csv", df)
````





Much better! `0.78947` on that submission.
