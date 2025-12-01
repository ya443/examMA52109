
# PACKAGE EXPLANATION

`cluster_maker` is a Python package that provides a complete workflow for clustering analysis, covering data creation, preparation, algorithm execution, evaluation, and visualisation.  
Its design emphasises clarity, reliability, and a logical end-to-end process suitable for demonstrations, testing, and analytical work.

# Key Capabilities

- Generate synthetic datasets with defined cluster structures.  
- Explore and summarise data through basic statistical analysis.  
- Prepare data for clustering through feature selection and standardisation.  
- Perform K-means clustering using either a simple manual implementation or the scikit-learn version.  
- Assess clustering quality with standard evaluation metrics and diagnostic plots.  
- Visualise clustering results and inertia curves clearly.  
- Execute the full clustering workflow in a single high-level function.

# dataframe_builder.py

## Purpose  
Creates synthetic datasets with predefined cluster structures for controlled experimentation.

## Key Functions  
- **define_dataframe_structure(column_specs)** – constructs a template containing cluster centres and feature definitions.  
- **simulate_data(seed_df, n_points, cluster_std, random_state)** – generates data points around each centre with added noise and a `true_cluster` label.

# data_analyser.py

## Purpose  
Provides descriptive insights to support initial exploration of the dataset.

## Key Functions  
- **calculate_descriptive_statistics(data)** – returns standard summary metrics.  
- **calculate_correlation(data)** – generates a correlation matrix for numeric features.

# data_exporter.py

## Purpose  
Supports the clean and controlled export of processed data or results.

## Key Functions  
- **export_to_csv(data, filename, delimiter, include_index)** – saves data to a CSV file.  
- **export_formatted(data, file, include_index)** – writes a readable, well-formatted table to a text file.

# preprocessing.py

## Purpose  
Prepares datasets for clustering by validating inputs and ensuring all features share a comparable scale.

## Key Functions  
- **select_features(data, feature_cols)** – checks that chosen features exist and are numeric.  
- **standardise_features(X)** – standardises all features using `StandardScaler`.

# algorithms.py

## Purpose  
Implements K-means clustering both manually and through scikit-learn for comparison and practical use.

## Key Functions  
- **init_centroids(X, k)** – selects initial centroids.  
- **assign_clusters(X, centroids)** – assigns each point to its nearest centroid.  
- **update_centroids(X, labels, k)** – recalculates centroid positions.  
- **kmeans(X, k)** – runs the full manual K-means routine.  
- **sklearn_kmeans(X, k)** – executes the scikit-learn implementation.

# evaluation.py

## Purpose  
Provides metrics that assess clustering structure and performance.

## Key Functions  
- **compute_inertia(X, labels, centroids)** – calculates within-cluster variance.  
- **silhouette_score_sklearn(X, labels)** – computes silhouette scores to measure cohesion and separation.  
- **elbow_curve(X, k_values, use_sklearn)** – evaluates inertia across k values for elbow analysis.

# plotting_clustered.py

## Purpose  
Generates visualisations that clarify clustering behaviour and support interpretation.

## Key Functions  
- **plot_clusters_2d(X, labels, centroids, title)** – produces a 2D scatter plot of clusters.  
- **plot_elbow(k_values, inertias, title)** – visualises inertia values for selecting an appropriate number of clusters.

# interface.py

## Purpose  
Provides a single high-level function that performs the complete clustering workflow.

## Key Function  
- **run_clustering(...)**  
  Loads and prepares data, applies the selected clustering method, computes evaluation metrics, generates visualisations, and saves outputs.  
  Returns a structured dictionary containing labelled data, metrics, centroids, and generated figures.

# Summary

`cluster_maker` integrates data generation, preprocessing, clustering, evaluation, and visualisation into a cohesive and transparent framework. It supports both controlled experimentation with synthetic data and practical clustering analysis, offering a clear end-to-end process while still allowing detailed examination of each individual step.


# EXPLANATION OF DEMO FILE AND THE CHANGES WITHIN IT

## demo file description (cluster_plot.py)

The `cluster_plot.py` demo script is designed to showcase how the `cluster_maker` package can be used to compare k-means clustering results across different values of *k*.  
Its purpose is to provide a simple, reproducible example of:

- loading a dataset from a CSV file,  
- selecting numeric features for clustering,  
- running k-means for several values of *k*,  
- producing 2D cluster plots,  
- computing evaluation metrics such as inertia and silhouette score, and  
- summarising results visually and in CSV form.

The script is intended to help users understand how clustering behaviour changes as the number of clusters varies.

## What was wrong with the original demo script, and how it was fixed

The original demo/cluster_plot.py script ran without crashing, but it **did not perform the intended analysis**. Although it printed results for k = 2, 3, 4, 5, it actually **did not run k-means with k = 4 or k = 5**, and it also **failed to generate the silhouette plot** because it looked for the wrong metric name.

### Key mistake 1: the value of k was never used correctly
Inside the loop, the original code had:

```python
result = run_clustering(..., k=min(k, 3), ...)
```

`min(k, 3)` overrides the intended value of k.
This means:
- **k = 2** → runs correctly
- **k = 3** → runs correctly
- **k = 4** → actually ran with `k = 3`
- **k = 5** → actually ran with `k = 3`

Because of this, the script pretended to analyse clustering for four different k-values, but in reality the results for k = 3, 4, and 5 were identical.
This produced misleading inertia/silhouette values, incorrect cluster plots, and an incorrect metrics summary.

#### Fix:
`k = min(k, 3)` was replaced with `k = k` so that the clustering algorithm uses the correct number of clusters in each iteration.

### Key mistake 2: the silhouette score was never plotted
The script attempted to generate a silhouette bar chart using this condition:
if "silhouette_score" in metrics_df.columns:

However, the run_clustering function stores the silhouette metric under the key:
"silhouette"
Because of the incorrect key name, the condition was never satisfied, and the silhouette plot was never created, even when valid silhouette values were available.

#### Fix:

The check was corrected to:

```python
if "silhouette" in metrics_df.columns:
```
and later silhouette is correctly called from the metrics DataFrame.

```python
plt.bar(metrics_df["k"], metrics_df["silhouette"])
```

This allows the silhouette plot to be generated and saved correctly.

### Additional minor improvement

An introductory message was added at the top of the script to explain what the demo does, improving clarity for the user.

### Result after fixing the issues

With these corrections, the script now: 

- properly runs k-means with k = 2, 3, 4, and 5
- generates different, correct inertia and silhouette scores
- correctly saves all four cluster plots
- correctly produces the silhouette comparison bar chart
- provides outputs that match the original intended behaviour
