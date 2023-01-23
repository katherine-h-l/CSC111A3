"""CSC111 Winter 2021 Assignment 3: Graphs, Recommender Systems, and Clustering (Part 3)

Instructions (READ THIS FIRST!)
===============================

This Python module contains the functions you'll write for determining *clusters* of vertices
in a graph.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking CSC111 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult our Course Syllabus.

This file is Copyright (c) 2021 David Liu and Isaac Waller.
The assignment was written by Katherine Luo and Stewart Chandler. 
"""
import random

from a3_part2_recommendations import WeightedGraph, load_weighted_review_graph


################################################################################
# Part 3, Q1
################################################################################
def create_book_graph(review_graph: WeightedGraph,
                      threshold: float = 0.05,
                      score_type: str = 'unweighted') -> WeightedGraph:
    """Return a book graph based on the given review_graph.

    The score_type parameter plays the same role as in WeightedGraph.get_similarity_score.

    The returned book graph has the following properties:
        1. Its vertex set is exactly the set of book vertices in review_graph
            (items are book titles).
        2. For every two distinct books b1 and b2, let s(b1, b2) be their similarity score,
            where score_type specifies which similarity score to use.

            - If s(b1, b2) > threshold, there is an edge between b1 and b2 in the book graph
              with weight equal to s(b1, b2). Unlike Part 2, these weights will be floats,
              not ints!
            - Otherwise, there is no edge between b1 and b2.

    Preconditions:
        - score_type in {'unweighted', 'strict'}
    """
    graph = WeightedGraph()

    for item1 in review_graph.get_all_vertices('book'):
        graph.add_vertex(item1, 'book')

        for item2 in graph.get_all_vertices():
            if item2 != item1:
                score = review_graph.get_similarity_score(item1, item2, score_type)
                if score > threshold:
                    graph.add_edge(item1, item2, score)

    return graph


################################################################################
# Part 3, Q2
################################################################################
def cross_cluster_weight(book_graph: WeightedGraph, cluster1: set, cluster2: set) -> float:
    """Return the cross-cluster weight between cluster1 and cluster2.

    See assignment handout for the definition of cross-cluster weight.

    Preconditions:
        - cluster1 != set() and cluster2 != set()
        - cluster1.isdisjoint(cluster2)
        - Every item in cluster1 and cluster2 is a vertex in book_graph
    """
    total_weight = 0
    for v1 in cluster1:
        for v2 in cluster2:
            total_weight += book_graph.get_weight(v1, v2)

    return total_weight / (len(cluster1) * len(cluster2))


################################################################################
# Part 3, Q3 (don't modify this code)
################################################################################
def find_clusters_random(graph: WeightedGraph, num_clusters: int) -> list[set]:
    """Return a list of <num_clusters> vertex clusters for the given graph.

    At each iteration, this algorithm first chooses a random cluster, and then chooses
    the cluster that has the highest cross-cluster weight to the randomly-chose cluster
    to merge.

    Preconditions:
        - num_clusters >= 1
    """
    # Each book starts in its own cluster
    clusters = [{book} for book in graph.get_all_vertices()]

    for _ in range(0, len(clusters) - num_clusters):
        print(f'{len(clusters)} clusters')

        c1 = random.choice(clusters)
        # Pick the best cluster to merge c1 into.
        best = -1
        best_c2 = None

        for c2 in clusters:
            if c1 is not c2:
                score = cross_cluster_weight(graph, c1, c2)
                if score > best:
                    best = score
                    best_c2 = c2

        best_c2.update(c1)
        clusters.remove(c1)

    return clusters


def find_clusters_greedy(graph: WeightedGraph, num_clusters: int) -> list[set]:
    """Return a list of <num_clusters> vertex clusters for the given graph.

    At each iteration, this algorithm chooses the pair of clusters with the highest
    cross-cluster weight to merge.

    Preconditions:
        - num_clusters >= 1
    """
    # Each book starts in its own cluster
    clusters = [{book} for book in graph.get_all_vertices()]

    for _ in range(0, len(clusters) - num_clusters):
        print(f'{len(clusters)} clusters')

        # Merge the two communities with the most links
        best = -1
        best_c1, best_c2 = None, None

        for i1 in range(0, len(clusters)):
            for i2 in range(i1 + 1, len(clusters)):
                c1, c2 = clusters[i1], clusters[i2]
                score = cross_cluster_weight(graph, c1, c2)
                if score > best:
                    best, best_c1, best_c2 = score, c1, c2

        best_c2.update(best_c1)
        clusters.remove(best_c1)

    return clusters


if __name__ == '__main__':
    # You can uncomment the following lines for code checking/debugging purposes.
    # However, we recommend commenting out these lines when working with the large
    # datasets, as checking representation invariants and preconditions greatly
    # increases the running time of the functions/methods.
    # import python_ta.contracts
    # python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'max-line-length': 1000,
        'disable': ['E1136'],
        'extra-imports': ['random', 'a3_part2_recommendations'],
        'allowed-io': ['find_clusters_greedy', 'find_clusters_random'],
        'max-nested-blocks': 4
    })
