import React from 'react';
import { useHistory } from '@docusaurus/router';

const StaticPageLink = ({ to, children }) => {
  const history = useHistory();

  const handleClick = (event) => {
    event.preventDefault();
    window.location.href = to;
  };

  return (
    <a href={to} onClick={handleClick}>
      {children}
    </a>
  );
};

export default StaticPageLink;
