export const DEFAULT_LANG = "en" as const;

export function getLangFromUrl(url: URL) {
  const [, lang] = url.pathname.split("/");
  if (lang in ["en", "es"]) return lang;
  return DEFAULT_LANG;
}

export function getTranslation(locale: any, textToTranslate: any) {
  const properTranslations = TRANSLATIONS[locale || DEFAULT_LANG];
  if (properTranslations[textToTranslate] == undefined) {
    return textToTranslate;
  }
  return properTranslations[textToTranslate];
}

export const TRANSLATIONS: { [key: string]: { [key: string]: string } } = {
  en: {
    Actualizado: "Updated",
    "Saltar al contenido": "Skip to content",
    Investigador: "Researcher",
    Profesor: "Professor",
    Músico: "Musician",
    Blog: "Blog",
    CV: "CV",
    Search: "Search",
    Prev: "Previous",
    Next: "Next",
    "Reflexiones...": "Reflections...",
    " result": " result",
    " results": " results",
    "Busca en el Blog...": "Search in the Blog...",
    "Comparte esta entrada en:": "Share this post in:",
    "Se encontró": "Found",
    para: "for",
    INVESTIGADOR: "RESEARCHER",
    PROFESOR: "PROFESSOR",
    MÚSICO: "MUSICIAN",
    YOUTUBER: "YOUTUBER",
    BLOG: "BLOG",
    "¿Qué estoy haciendo?": "NEWS",
    "Todas las entradas": "All posts",
    '"El todo es mucho menos que la suma de mis partes"':
      '"The whole is much less than the sum of my parts"',
    "All rights reserved.": "All rights reserved",
    "Página no encontrada": "Page not found",
    "Not Found": "Not Found",
    "Go back": "Go back",
    "All the articles with the tag": "All the articles with the tag",
    "Volver al principio": "Back to top",
    "Ir atrás": "Go back",
    "Te convertirás en un experto en la extracción de datos de la web. Este es el curso más completo sobre Web Scraping de toda la Internet. Te enseño desde CERO los fundamentos del Web Scraping de una manera muy sencilla de entender; realizando extracción de datos de más de 20 páginas web":
      "You will become an expert in Web Scraping. This is the most complete course about Web Scraping on the entire Internet. I will teach you from 0 the fundamentals of Web Scraping in a very easy to understand way; performing data extraction from more than 20 web pages",
    horas: "hours",
    clases: "lectures",
    alumnos: "students",
    reseñas: "reviews",
    "¡ESCRÍBEME POR CUPONES!": "PING ME FOR COUPONS!",
    "Visualizaciones y Análisis de Datos": "Visualizations and Data Analysis",
    "Cada clase es un nuevo desafío que tendremos que resolver con el poder del análisis de datos. Te enseño desde CERO los principios de +45 visualizacioens diferentes y del análisis de datos de una manera muy sencilla de entender.":
      "Each lecture proposes a new challenge that we will have to solve with the power of data analysis. I will teach you from 0 the principles of +45 different visualizations and the principles of data analysis in a very easy to understand way.",
    "Escrito por": "Written by",
    "Tiempo de lectura:": "Time to read:",
    minutos: "min",
  },
  es: {
    Actualizado: "Actualizado",
    "Saltar al contenido": "Saltar al contenido",
    Investigador: "Investigador",
    Profesor: "Profesor",
    Músico: "Músico",
    Blog: "Blog",
    CV: "CV",
    Search: "Búsqueda",
    Prev: "Anterior",
    Next: "Siguiente",
    "Reflexiones...": "Reflexiones...",
    " result": " resultado",
    " results": " resultados",
    "Busca en el Blog...": "Busca en el Blog...",
    "Comparte esta entrada en:": "Comparte esta entrada en:",
    "Se encontró": "Se encontró",
    para: "para",
    INVESTIGADOR: "INVESTIGADOR",
    PROFESOR: "PROFESOR",
    MÚSICO: "MÚSICO",
    YOUTUBER: "YOUTUBER",
    BLOG: "BLOG",
    "¿Qué estoy haciendo?": "¿Qué estoy haciendo?",
    "Todas las entradas": "Todas las entradas",
    '"El todo es mucho menos que la suma de mis partes"':
      '"El todo es mucho menos que la suma de mis partes"',
    "All rights reserved.": "Todos los derechos reservados",
    "Página no encontrada": "Página no encontrada",
    "Not Found": "No encontrada",
    "Go back": "Regresar",
    "All the articles with the tag": "Todos los artículos con el tag",
    "Volver al principio": "Volver al principio",
    "Ir atrás": "Go back",
    "Te convertirás en un experto en la extracción de datos de la web. Este es el curso más completo sobre Web Scraping de toda la Internet. Te enseño desde CERO los fundamentos del Web Scraping de una manera muy sencilla de entender; realizando extracción de datos de más de 20 páginas web":
      "Te convertirás en un experto en la extracción de datos de la web. Este es el curso más completo sobre Web Scraping de toda la Internet. Te enseño desde CERO los fundamentos del Web Scraping de una manera muy sencilla de entender; realizando extracción de datos de más de 20 páginas web",
    horas: "horas",
    clases: "clases",
    alumnos: "alumnos",
    reseñas: "reseñas",
    "¡ESCRÍBEME POR CUPONES!": "¡ESCRÍBEME POR CUPONES!",
    "Visualizaciones y Análisis de Datos":
      "Visualizaciones y Análisis de Datos",
    "Cada clase es un nuevo desafío que tendremos que resolver con el poder del análisis de datos. Te enseño desde CERO los principios de +45 visualizacioens diferentes y del análisis de datos de una manera muy sencilla de entender.":
      "Cada clase es un nuevo desafío que tendremos que resolver con el poder del análisis de datos. Te enseño desde CERO los principios de +45 visualizacioens diferentes y del análisis de datos de una manera muy sencilla de entender.",
    "Escrito por": "Escrito por",
    "Tiempo de lectura:": "Tiempo de lectura:",
    minutos: "min",
  },
} as const;
