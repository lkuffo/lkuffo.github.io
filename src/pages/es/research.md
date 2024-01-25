---
layout: ../../layouts/ResearchLayout.astro
title: ""
---

<h1 class="title-box" > INVESTIGACIÓN </h1>

<div class="text-sm text-red-600 text-center w-full"> Psst: Voy a estar en SIGMOD 2024 en Santiago, Chile! </div>

¡Me gusta hacer a las Bases de Datos más rápidas y que utilicen menos almacenamiento! Recientemente he estado trabajando en **Compresión de Datos** y **Vector Similarity Search** (ANN) dentro de Bases de Datos de Vectores en conjunto con el Profesor [Peter Boncz](https://homepages.cwi.nl/~boncz/) en [CWI](https://www.cwi.nl/en/) (Centrum Wiskunde & Informatica) -- donde me encuentro actualmente haciendo mi PhD.

Antes, trabajé haciendo investigación en Social Data Science y Opinion Mining con la Profesora Carmen Vaca en la Universidad ESPOL.

<span>[Google Scholar](https://scholar.google.com/citations?user=g6vekOwAAAAJ) <svg class="icon-tabler" fill="#000000" width="800px" height="800px" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
<path d="M14.573 2.729c-0.729 0.484-4.292 2.849-7.917 5.255s-6.589 4.396-6.589 4.422c0 0.026 0.182 0.146 0.406 0.266 0.224 0.13 3.797 2.109 7.953 4.411l7.542 4.193 0.193-0.099c0.109-0.052 2.891-1.641 6.188-3.521l5.99-3.427 0.036 10.599h3.557v-12.401l-4.615-3.094c-6.219-4.167-11.188-7.448-11.307-7.474-0.063-0.010-0.703 0.38-1.438 0.87zM7.141 22.177l0.016 2.672 8.828 5.292 8.891-5.339v-2.641c0-1.458-0.016-2.646-0.031-2.646-0.021 0-1.76 1.042-3.87 2.323l-4.406 2.661-0.578 0.339-1.755-1.052c-1.464-0.875-2.927-1.755-4.385-2.641l-2.672-1.615c-0.031-0.010-0.042 1.177-0.036 2.646z"/>
</svg></span>

<h2 style="text-align: center;" class="title-box !mb-8"> PUBLICACIONES RECIENTES </h2>

<span class="text-skin-accent text-mono text-lg font-semibold">ALP: Adaptive Lossless floating-Point Compression</span>  
<sup class="text-gray-800"> [2023] </sup> <sup class="text-gray-800">Afroozeh, Azim, Leonardo Kuffo, and Peter Boncz </sup>  
ALP es un algoritmo de compresión de datos de tipo `float` y `double` con una altísima velocidad de descompresión. Diseñado para aprovechar el esquema de almacenamientos de Bases de Datos columnares. ALP comprime mejor y es más rápido que Chimp128, Patas, Elf y Zstd. ALP le debe su velocidad a su simpleza, su diseño vectorizado (opera en 1024 valores a la vez) y al uso de instrucciones SIMD. ALP se encuentra en producción dentro de [DuckDB](https://duckdb.org/); en donde contribuí para la implementación.

<span class="text-skin-accent text-mono text-lg font-semibold">Mining worldwide entrepreneurs psycholinguistic dimensions from Twitter</span>  
<sup class="text-gray-800"> [2018] </sup> <sup class="text-gray-800">L Kuffó, C Vaca, E Izquierdo, JC Bustamante</sup>  
En este estudio caracterizamos a emprendedores de diversas partes del mundo a través del lenguaje que utilizaban para comunicarse en una red social. Logramos correlacionar las emociones dentro del discurso de los emprendedores con varios índices de emprendimiento.

<span class="text-skin-accent text-mono text-lg font-semibold">Back to# 6D: Predicting Venezuelan states political election results through Twitter</span>  
<sup class="text-gray-800"> [2017] </sup> <sup class="text-gray-800">R Castro, L Kuffó, C Vaca </sup>  
Back to #6D propone un framework para hacer estimaciones de resultados electorales a gran escala utilizando una red social en donde las personas expresen sus opiniones. En este estudio utilizamos netamente datos de Twitter para estimar los resultados de las elecciones parlamentarias de Venezuela ocurridas en 2016.
