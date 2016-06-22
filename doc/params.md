KGraph Parameter Tuning
=======================
KGraph supports a number of parameters for both indexing and
searching so as to fine tune the performance.

In the C++ API, these parameters are passed in as fields of the
KGraph::IndexParams and KGraph::SearchParams.
The constructors of these two structs sets all the parameters
to default values that will work reasonably well for many datasets.

In the Python API, these parameters are passed in as optional
keyword arguments to the build and search methods.  When not set,
the same default values are assumed.

# General Guidelines for Parameter Tuning

## Online k-NN Search

`SearchParams::K` should be determined by the application.
Enlarging any of `P, M, S, T` has the effect of increasing recall and
slowing down speed at the same time.

Enlarging `P` is the primary way of increasing
recall at the cost of slowing down speed.

If the index was created with `reverse` = 0, changing `M` between
`IndexParams::K` and `IndexParams::L` is the secondary way of
tuning accuracy and speed.

Enlarging `T` typically does not work.  The effect of `S` is typically
not significant.

The online search parameters always change accuracy and speed at the
same time.  To speed up search without sacrificing accuracy, one has
to re-construct the index, typically with a larger `IndexParams::K`
and larger `IndexParams::L`, as discussed below.

## Indexing/k-NN Graph Construction

For indexing purpose, it is always recommended to set `reverse` to -1.
If the goal is to extract the k-NN graph, then `reserse` has to be 0.

Increasing any of `K`, `L`, `S` and `R` has the effect of improving
accuracy and slowing down speed at the same time.

In a simplified view, KGraph constructs a M-NN graph, with K <= M <= L,
and M being different for each object depending on its local intrinsinc
dimension.  The M-NN graph, with varying M, is also the actual index
when `reverse` is set to 0 or -1.  According to this, `K` is the lower-bound
of the per-object cost and `L` is the upper-bound, with KGraph to freely
pick a suitable value between `K` and `L` for each object.
`L` is set to at least `K + 50` to give KGraph some wiggling space.
Typical settings are (K = 25, L = 100), (K=50~100, L=150), (K=200, L=300), etc.

Enlarging `S` slightly increases accuracy, but slows down computation significantly,
and is typically set below 30.
`R` typically does not have to be changed.


# Index Parameters

| Name       | Default | Description |
|------------|---------|-------------|
| K          | 25      |             |
| L          | 100     | >= K + 50   |
| S          | 10      | Use default.|
| R          | 100     | Use default.|
| iterations | 30      | See 2.      |
| controls   | 100     | See 1.      |
| recall     | 0.99    | See 1, 2.   |
| delta      | 0.002   | See 2.      |
| reverse    | 0       | See 3.      |
| seed       | 1998    | Random seed.|

## 1. On-the-fly accuracy estimation

When constructing a k-NN graph, KGraph estimate its accuracy
after each iteration, and stops iterating when the estimated
accuracy exceeds the given `recall`.

For this purpose, KGraph randomly sample a number of control
points, whose k-NN are found with brutal force search.  The
number of control points can be adjusted with the parameter
`controls`, but this is typically not needed.

Accuracy is measured in recall, which is the number of k-NN
actually found divided by k, averaged across all query objects.

## 2. Iteration stop criteria

KGraph stops iteration when at least one of the following
criteria is met:

* Number of iterations reach `iterations`.
* Estimated recall exceeds `recall`.
* Number of entries updated becomes less than `delta`*K*N.

## 3. Adding Reverse Edges

For indexing purpose, it usually helps to add the reverse edges of the k-NN
graph.  This can be enabled by setting `reverse` to a non-zero value.  If
`reverse` is set to a positive value `Kp`, then the graph is first trimmed from
the original `K`-NN graph to a `Kp`-NN graph, and all reverse edges are added.

If `reverse` is set to -1, the recommended setting for indexing purpose,
the graph is automatically trimmed to a suitable size, and then all reverse
edges are added.

# Search Parameters

| Name       | Default | Description |
|------------|---------|-------------|
| K          | 25      | Desired K.  |
| M          | 0       | Use default.|
| P          | 100     | See 1.      |
| S          | 10      | Use default.|
| T          | 1       | See 1.      |
| epsilon    | +1e30   | See 2.      |
| init       | 0       | See 3.      |
| seed       | 1998    | Random seed.|

## 1. Computation and Accuracy

`P` is the main parameter to control the amount of computation.
Increase `P` will leads to higher recall as well as more computation.
The same search process is repeated `T` times with results merged.
Typically there is no need for a `T` > 1, but for some datasets it helps.

## 2. epsilon-NN search

Entries of a similarity value bigger than `epsilon` are removed, so there may be
less than `K` items returned.

## 3. User-Provided Starting Points

KGraph can the index to refine k-NN search results obtained from another algorithm.
To use this, set `init` to the number of initial k-NN items, and pass in the items
via the `ids` parameter of KGraph::search.  The input items in the buffer are
overwritten when search is done.

