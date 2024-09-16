// Swizzled class to show custom text for canary version.
// Should be removed in favor of the stock implementation.

import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import {ThemeClassNames} from '@docusaurus/theme-common';
import { useLocalPathname } from '@docusaurus/theme-common/internal';
function UnreleasedVersionLabel({siteTitle, versionMetadata}) {
  return (
    <Translate
      id="theme.docs.versions.unreleasedVersionLabel"
      description="The label used to tell the user that he's browsing an unreleased doc version"
      values={{
        siteTitle,
        versionLabel: <b>{versionMetadata.label}</b>,
      }}>
      {
        'This is unreleased documentation for {siteTitle}\'s {versionLabel} version.'
      }
    </Translate>
  );
}
function UnmaintainedVersionLabel({siteTitle, versionMetadata}) {
  return (
    <Translate
      id="theme.docs.versions.unmaintainedVersionLabel"
      description="The label used to tell the user that he's browsing an unmaintained doc version"
      values={{
        siteTitle,
        versionLabel: <b>{versionMetadata.label}</b>,
      }}>
      {
        'This is documentation for {siteTitle} {versionLabel}, which is no longer actively maintained.'
      }
    </Translate>
  );
}
const BannerLabelComponents = {
  unreleased: UnreleasedVersionLabel,
  unmaintained: UnmaintainedVersionLabel,
};
function BannerLabel(props) {
  const BannerLabelComponent =
    BannerLabelComponents[props.versionMetadata.banner];
  return <BannerLabelComponent {...props} />;
}
function LatestVersionSuggestionLabel({versionLabel, to, onClick}) {
  return (
    <Translate
      id="theme.docs.versions.latestVersionSuggestionLabel"
      description="The label used to tell the user to check the latest version"
      values={{
        versionLabel,
        latestVersionLink: (
          <b>
            <Link to={to} onClick={onClick}>
              <Translate
                id="theme.docs.versions.latestVersionLinkLabel"
                description="The label used for the latest version suggestion link label">
                this version
              </Translate>
            </Link>
          </b>
        ),
      }}>
      {
        'For the current stable version, see {latestVersionLink} ({versionLabel}).'
      }
    </Translate>
  );
}

export default function DocVersionBanner({className}) {
  const versionMetadata = {
    badge: false,
    banner: 'unmaintained',
    isLast: false,
    label: 'v0.1',
    noIndex: false,
    pluginId: 'default',
    version: 'Latest',
  }
  console.log({versionMetadata});
  const localPathname = useLocalPathname();
  if (versionMetadata.banner) {
    return (
      <div
        className={clsx(
          className,
          ThemeClassNames.docs.docVersionBanner,
          'alert alert--warning margin-bottom--md',
        )}
        role="alert">
        <div>
          <BannerLabel siteTitle={"LangChain"} versionMetadata={versionMetadata} />
        </div>
        <div className="margin-top--md">
          <LatestVersionSuggestionLabel
            versionLabel={"Latest"}
            to={`https://python.langchain.com${localPathname}`}
            onClick={() => {}}
          />
        </div>
      </div>
    );
  }
  return null;
}
