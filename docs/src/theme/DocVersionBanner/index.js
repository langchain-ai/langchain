// Swizzled class to show custom text for canary version.
// Should be removed in favor of the stock implementation.

import React from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import Translate from '@docusaurus/Translate';
import {
  useActivePlugin,
  useDocVersionSuggestions,
} from '@docusaurus/plugin-content-docs/client';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {
  useDocsPreferredVersion,
  useDocsVersion,
} from '@docusaurus/theme-common/internal';
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
function DocVersionBannerEnabled({className, versionMetadata}) {
  const {
    siteConfig: {title: siteTitle},
  } = useDocusaurusContext();
  const {pluginId} = useActivePlugin({failfast: true});
  const getVersionMainDoc = (version) =>
    version.docs.find((doc) => doc.id === version.mainDocId);
  const {savePreferredVersionName} = useDocsPreferredVersion(pluginId);
  const {latestDocSuggestion, latestVersionSuggestion} =
    useDocVersionSuggestions(pluginId);
  // Try to link to same doc in latest version (not always possible), falling
  // back to main doc of latest version
  const latestVersionSuggestedDoc =
    latestDocSuggestion ?? getVersionMainDoc(latestVersionSuggestion);
  return (
    <div
      className={clsx(
        className,
        ThemeClassNames.docs.docVersionBanner,
        'alert alert--warning margin-bottom--md',
      )}
      role="alert">
      <div>
        <BannerLabel siteTitle={siteTitle} versionMetadata={versionMetadata} />
      </div>
      <div className="margin-top--md">
        <LatestVersionSuggestionLabel
          versionLabel={latestVersionSuggestion.label}
          to={latestVersionSuggestedDoc.path}
          onClick={() => savePreferredVersionName(latestVersionSuggestion.name)}
        />
      </div>
    </div>
  );
}

function LatestDocVersionBanner({className, versionMetadata}) {
  const {
    siteConfig: {title: siteTitle},
  } = useDocusaurusContext();
  const {pluginId} = useActivePlugin({failfast: true});
  const getVersionMainDoc = (version) =>
    version.docs.find((doc) => doc.id === version.mainDocId);
  const {savePreferredVersionName} = useDocsPreferredVersion(pluginId);
  const {latestDocSuggestion, latestVersionSuggestion} =
    useDocVersionSuggestions(pluginId);
  // Try to link to same doc in latest version (not always possible), falling
  // back to main doc of latest version
  const latestVersionSuggestedDoc =
    latestDocSuggestion ?? getVersionMainDoc(latestVersionSuggestion);
  const canaryPath = `/docs/0.2.x/${latestVersionSuggestedDoc.path.slice("/docs/".length)}`;
  return (
    <div
      className={clsx(
        className,
        ThemeClassNames.docs.docVersionBanner,
        'alert alert--info margin-bottom--md',
      )}
      role="alert">
      <div>
        <Translate
          id="theme.docs.versions.unmaintainedVersionLabel"
          description="The label used to encourage the user to view the experimental 0.2.x version"
          values={{
            siteTitle,
            versionLabel: <b>{versionMetadata.label}</b>,
          }}>
          {
            'This is a stable version of documentation for {siteTitle}\'s version {versionLabel}.'
          }
        </Translate>
      </div>
      <div className="margin-top--md">
        <Translate
          id="theme.docs.versions.latestVersionSuggestionLabel"
          description="The label used to tell the user to check the experimental version"
          values={{
            versionLabel: <b>{versionMetadata.label}</b>,
            latestVersionLink: (
              <b>
                <Link to={canaryPath} onClick={() => savePreferredVersionName("0.2.x")}>
                  <Translate
                    id="theme.docs.versions.latestVersionLinkLabel"
                    description="The label used for the latest version suggestion link label">
                    this experimental version
                  </Translate>
                </Link>
              </b>
            ),
          }}>
          {
            'You can also check out {latestVersionLink} for an updated experience.'
          }
        </Translate>
      </div>
    </div>
  );
}

export default function DocVersionBanner({className}) {
  const versionMetadata = useDocsVersion();
  if (versionMetadata.banner) {
    return (
      <DocVersionBannerEnabled
        className={className}
        versionMetadata={versionMetadata}
      />
    );
  } else if (versionMetadata.isLast) {
    // Uncomment when we are ready to direct people to new build
    // return (
    //   <LatestDocVersionBanner
    //     className={className}
    //     versionMetadata={versionMetadata}
    //   />
    // );
    return null;
  }
  return null;
}
