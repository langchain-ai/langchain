import React from 'react';
import clsx from 'clsx';
import Container from '@theme/CodeBlock/Container';
import styles from './styles.module.css';
// <pre> tags in markdown map to CodeBlocks. They may contain JSX children. When
// the children is not a simple string, we just return a styled block without
// actually highlighting.
export default function CodeBlockJSX({children, className}) {
  return (
    <Container
      as="pre"
      tabIndex={0}
      className={clsx(styles.codeBlockStandalone, 'thin-scrollbar', className)}>
      <code className={styles.codeBlockLines}>{children}</code>
    </Container>
  );
}
