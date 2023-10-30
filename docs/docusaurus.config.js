/* eslint-disable global-require,import/no-extraneous-dependencies */

// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
// eslint-disable-next-line import/no-extraneous-dependencies
const { ProvidePlugin } = require("webpack");
const path = require("path");

const baseLightCodeBlockTheme = require("prism-react-renderer/themes/vsLight");
const baseDarkCodeBlockTheme = require("prism-react-renderer/themes/vsDark");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "🦜️🔗 Langchain",
  tagline: "LangChain Python Docs",
  favicon: "img/favicon.ico",
  customFields: {
    mendableAnonKey: process.env.MENDABLE_ANON_KEY,
  },
  // Set the production url of your site here
  url: "https://python.langchain.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "throw",

  plugins: [
    () => ({
      name: "custom-webpack-config",
      configureWebpack: () => ({
        plugins: [
          new ProvidePlugin({
            process: require.resolve("process/browser"),
          }),
        ],
        resolve: {
          fallback: {
            path: false,
            url: false,
          },
        },
        module: {
          rules: [
            {
              test: /\.m?js/,
              resolve: {
                fullySpecified: false,
              },
            },
            {
              test: /\.py$/,
              loader: "raw-loader",
              resolve: {
                fullySpecified: false,
              },
            },
            {
              test: /\.ipynb$/,
              loader: "raw-loader",
              resolve: {
                fullySpecified: false,
              },
            },
          ],
        },
      }),
    }),
  ],

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          remarkPlugins: [
            [require("@docusaurus/remark-plugin-npm2yarn"), { sync: true }],
          ],
          async sidebarItemsGenerator({
            defaultSidebarItemsGenerator,
            ...args
          }) {
            const sidebarItems = await defaultSidebarItemsGenerator(args);
            sidebarItems.forEach((subItem) => {
              // This allows breaking long sidebar labels into multiple lines
              // by inserting a zero-width space after each slash.
              if (
                "label" in subItem &&
                subItem.label &&
                subItem.label.includes("/")
              ) {
                // eslint-disable-next-line no-param-reassign
                subItem.label = subItem.label.replace(/\//g, "/\u200B");
              }
            });
            return sidebarItems;
          },
        },
        pages: {
          remarkPlugins: [require("@docusaurus/remark-plugin-npm2yarn")],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: true,
        },
      },
      colorMode: {
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      prism: {
        theme: {
          ...baseLightCodeBlockTheme,
          plain: {
            ...baseLightCodeBlockTheme.plain,
            backgroundColor: "#F5F5F5",
          },
        },
        darkTheme: {
          ...baseDarkCodeBlockTheme,
          plain: {
            ...baseDarkCodeBlockTheme.plain,
            backgroundColor: "#222222",
          },
        },
      },
      image: "img/parrot-chainlink-icon.png",
      navbar: {
        title: "🦜️🔗 LangChain",
        items: [
          {
            to: "/docs/get_started/introduction",
            label: "Docs",
            position: "left",
          },
          {
            type: "docSidebar",
            position: "left",
            sidebarId: "use_cases",
            label: "Use cases",
          },
          {
            type: "docSidebar",
            position: "left",
            sidebarId: "integrations",
            label: "Integrations",
          },
          {
            href: "https://api.python.langchain.com",
            label: "API",
            position: "left",
          },
          {
            type: "dropdown",
            label: "More",
            position: "left",
            items: [
              {
                to: "/docs/community",
                label: "Community",
              },
              {
                to: "/docs/contributing",
                label: "Developer's guide",
              },
              {
                to: "/docs/additional_resources/dependents",
                label: "Dependents",
              },
              {
                to: "/docs/additional_resources/tutorials",
                label: "Tutorials"
              },
              {
                to: "/docs/additional_resources/youtube",
                label: "YouTube videos"
              },
              { label: "Gallery", href: "https://github.com/kyrolabs/awesome-langchain" }
            ]
          },
          {
            type: "dropdown",
            label: "Also by LangChain",
            position: "right",
            items: [
              {
                href: "https://chat.langchain.com",
                label: "Chat our docs",
              },
              {
                href: "https://smith.langchain.com",
                label: "LangSmith",
              },
              {
                href: "https://smith.langchain.com/hub",
                label: "LangChain Hub",
              },
              {
                href: "https://github.com/langchain-ai/langserve",
                label: "LangServe",
              },
              {
                href: "https://js.langchain.com/docs",
                label: "JS/TS",
              },
            ]
          },
          // Please keep GitHub link to the right for consistency.
          {
            href: "https://github.com/langchain-ai/langchain",
            position: "right",
            className: "header-github-link",
            "aria-label": "GitHub repository",
          },
        ],
      },
      footer: {
        style: "light",
        links: [
          {
            title: "Community",
            items: [
              {
                label: "Discord",
                href: "https://discord.gg/cU2adEyC7w",
              },
              {
                label: "Twitter",
                href: "https://twitter.com/LangChainAI",
              },
            ],
          },
          {
            title: "GitHub",
            items: [
              {
                label: "Python",
                href: "https://github.com/langchain-ai/langchain",
              },
              {
                label: "JS/TS",
                href: "https://github.com/langchain-ai/langchainjs",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "Homepage",
                href: "https://langchain.com",
              },
              {
                label: "Blog",
                href: "https://blog.langchain.dev",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} LangChain, Inc.`,
      },
    }),

  scripts: [
    "/js/google_analytics.js",
    {
      src: "https://www.googletagmanager.com/gtag/js?id=G-9B66JQQH2F",
      async: true,
    },
  ],
};

module.exports = config;
