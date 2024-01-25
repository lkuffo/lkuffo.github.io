---
layout: ../../layouts/ResearchLayout.astro
title: ""
---

<h1 class="title-box" > RESEARCH </h1>

<div class="text-sm text-red-600 text-center w-full"> Psst: I am going to be on SIGMOD 2024 in Santiago, Chile! Let's get in touch! (ping me) </div>

I like to make databases faster and more efficient! Recently, I have been working on **Data Compression** and **Vector Similarity Search** (ANN) within Vector Databases. I am currently working with Prof. [Peter Boncz](https://homepages.cwi.nl/~boncz/) at [CWI](https://www.cwi.nl/en/) as part of the Database Architectures group -- where I am doing a PhD.

Before, I worked doing research in **Social Data Science** and **Opinion Mining** with Prof. Carmen Vaca at ESPOL University (Guayaquil, Ecuador)

<span>[Google Scholar](https://scholar.google.com/citations?user=g6vekOwAAAAJ) <svg class="icon-tabler" fill="#000000" width="800px" height="800px" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
<path d="M14.573 2.729c-0.729 0.484-4.292 2.849-7.917 5.255s-6.589 4.396-6.589 4.422c0 0.026 0.182 0.146 0.406 0.266 0.224 0.13 3.797 2.109 7.953 4.411l7.542 4.193 0.193-0.099c0.109-0.052 2.891-1.641 6.188-3.521l5.99-3.427 0.036 10.599h3.557v-12.401l-4.615-3.094c-6.219-4.167-11.188-7.448-11.307-7.474-0.063-0.010-0.703 0.38-1.438 0.87zM7.141 22.177l0.016 2.672 8.828 5.292 8.891-5.339v-2.641c0-1.458-0.016-2.646-0.031-2.646-0.021 0-1.76 1.042-3.87 2.323l-4.406 2.661-0.578 0.339-1.755-1.052c-1.464-0.875-2.927-1.755-4.385-2.641l-2.672-1.615c-0.031-0.010-0.042 1.177-0.036 2.646z"/>
</svg></span>

<h2 style="text-align: center;" class="title-box !mb-8"> RECENT PUBLICATIONS </h2>

<span class="text-skin-accent text-mono text-lg font-semibold">ALP: Adaptive Lossless floating-Point Compression</span>  
<sup class="text-gray-800"> [2023] </sup> <sup class="text-gray-800">Afroozeh, Azim, Leonardo Kuffo, and Peter Boncz </sup>  
ALP is an algorithm to compress `float` and `double` data; with an insane decompression speed (sub-cycle performance per tuple). ALP can compress better and faster than Chimp128, Patas, ELF and even Zstd. ALP speed stems from its simple algorithm, its vectorized design (operating on 1024 tuples at a time) and the use of SIMD instructions. ALP is currently being used inside [DuckDB](https://duckdb.org/); for which I contributed to [the implementation](https://github.com/duckdb/duckdb/pull/9635).

<span class="text-skin-accent text-mono text-lg font-semibold">Mining worldwide entrepreneurs psycholinguistic dimensions from Twitter</span>  
<sup class="text-gray-800"> [2018] </sup> <sup class="text-gray-800">L Kuffó, C Vaca, E Izquierdo, JC Bustamante</sup>  
This study characterized entrepreneurs from different countries based on the emotions of their speech communicated through social media posts. We found strong correlations between the emotions within the entrepreneurs' posts and various entrepreneurship health indexes within their countries.

<span class="text-skin-accent text-mono text-lg font-semibold">Back to# 6D: Predicting Venezuelan states political election results through Twitter</span>  
<sup class="text-gray-800"> [2017] </sup> <sup class="text-gray-800">R Castro, L Kuffó, C Vaca </sup>  
Back to #6D proposes a framework to make estimations of election results at a country scale by analyzing people's opinions in a social network. In this study, we tested our framework by estimating the results of the Venezuelan Parliamentary Elections (2016) using solely Twitter posts published before the voting day.
