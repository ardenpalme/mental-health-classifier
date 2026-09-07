# mlp-project
MLP 2026 Group Project
<br>
The gray box in the project description has the most important instructions.

1. The project description says:
> You should start from an interpretable baseline model of your choice, including as few or as many of the provided variables.
so we should train a Logistic Regressoin model on the provided subset of features in `unicef_malawi.csv`
<br>

2. This is the objective of the project, and if we're using PCA then we'll need some way to map back to variables (perhaps need to consider no PCA?)
> Thus, your report should focus on describing and motivating your final model choice, along with a comparison against the baseline model. It is important that any interpretations and conclusions you draw from your model are well supported and sound and that you understand limitations of the model and the data.

3. Please make sure that all figures have analysis.
> if a figure is not explicitly discussed in the text it should not be in the final document


Triptesh, could you plz try training logistic regression on this reduced feature set, curious if removing all of these is useful
```
%%capture
features = get_features(['FCF', 'CL', 'FCD', 'PR', 'HC'], excl_features+['FCF17__FCF17', 'FCF21__FCF21', 'FCF8__FCF8', 'FCF19__FCF19'])
print(f"Total number of features: {len(features)}")
pre_processor = PreProcessor(df,features, 
                            num_features=list(set(num_features).intersection(set(features))), 
                            one_hot_features=list(set(one_hot_features).intersection(set(features))))could 

pipe = Pipeline([
    ('pre', pre_processor1),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=random_state)),
])

pipe.fit_transform(df[features])
```
