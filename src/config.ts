import type { Site, SocialObjects } from "./types";

export const SITE: Site = {
  website: "https://lkuffo.github.io/", // replace this with your deployed domain
  author: "Leonardo Kuffó",
  desc: "Hola, soy Leonardo Kuffó, y esto es lo que hago!",
  title: "Leonardo Kuffó",
  ogImage: "/assets/circular.png",
  lightAndDarkMode: true,
  postPerPage: 10,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
};

export const LOCALE = {
  lang: "en", // html lang code. Set this empty and default will be "en"
  langTag: ["en-EN"], // BCP 47 Language Tags. Set this empty [] to use the environment default
} as const;

export const LOGO_IMAGE = {
  enable: false,
  svg: true,
  width: 216,
  height: 46,
};

export const SOCIALS: SocialObjects = [
  {
    name: "YouTube",
    href: "https://www.youtube.com/kufgal",
    linkTitle: `${SITE.title} on YouTube`,
    active: true,
  },
  {
    name: "Udemy",
    href: "https://www.udemy.com/user/leonardo-xavier-kuffo-rivero/",
    linkTitle: `${SITE.title} on Udemy`,
    active: true,
  },
  {
    name: "Github",
    href: "https://github.com/lkuffo/",
    linkTitle: ` ${SITE.title} on Github`,
    active: true,
  },
  {
    name: "Instagram",
    href: "https://instagram.com/leo.profesor",
    linkTitle: `${SITE.title} on Instagram`,
    active: true,
  },
  {
    name: "LinkedIn",
    href: "https://linkedin.com/in/lkuffo",
    linkTitle: `${SITE.title} on LinkedIn`,
    active: true,
  },
  {
    name: "Mail",
    href: "kufgal@gmail.com",
    linkTitle: `Send an email to ${SITE.title}`,
    active: true,
  },
];
