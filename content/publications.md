---
title: "My Research"
layout: "publications"
url: "/publications/"
summary: publications
---

> **TL;DR**: I created [PDX](https://github.com/cwida/PDX), a vertical data layout that accelerates vector similarity search, and [ALP](https://github.com/cwida/ALP), the state-of-the-art in lossless floating-point data compression.

## PDX: A Vertical Data Layout for Vector Similarity Search
[PDX](https://github.com/cwida/PDX) is a **vertical** data layout for vectors that store the dimensions of different vectors together. In a nutshell, PDX is PAX for vector similarity search. PDX brings to the table:   
- Efficient pruning of dimensions with partial distance calculations. 
- Up to **7x faster** IVF queries than FAISS+AVX512.
- Up to **13x faster** exhaustive search thanks to pruning.
- Distance kernels are up to **1.6x faster** than the `float32` kernels in USearch/FAISS. 
- Distance kernels on small vectors are up to **8x faster** than SIMD kernels in [SimSIMD](https://github.com/ashvardanian/SimSIMD).

## ALP: Adaptive Lossless floating-Point Compression
[ALP](https://github.com/cwida/ALP) is an algorithm that compresses `float` and `double` data with an insane decompression speed (sub-cycle performance per tuple). ALP can compress better and faster than Chimp128, Patas, ELF, and even Zstd. ALP speed stems from its simple algorithm and its vectorized design (operates on 1024 tuples at a time). ALP is currently being used inside [DuckDB](https://duckdb.org/), for which I contributed [the implementation](https://github.com/duckdb/duckdb/pull/9635). Other systems that implement ALP are KuzuDB and SpiralDB, and many others are on their way to implementing ALP. There is also a [YouTube video](https://www.youtube.com/watch?v=GKG5b04o5Yc) of me presenting the details of the algorithm.

## Current Projects (looking for MSc. students!)
I am working on the [PDX](https://github.com/cwida/PDX) layout to turn it into a fully fledged library. We need to: (i) Support quantized vectors. (ii) Support predicated queries. (iii) Support HNSW indexes.   

If you are an MSc. student in the EU looking for thesis projects and are interested in Vector Similarity Search, you can apply to CWI, Amsterdam, by writing to lxkr@cwi.nl. If accepted, you will be supervised by the one and only [Peter Boncz](https://homepages.cwi.nl/~boncz/).