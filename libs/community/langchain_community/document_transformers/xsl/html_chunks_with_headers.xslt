<?xml version="1.0" encoding="UTF-8" ?>
<!-- HTML PRE CHUNK:
This performs a best-effort preliminary "chunking" of text in an HTML file,
matching each chunk with a "headers" metadata value based on header tags in proximity.

recursively visits every element (template mode=list).
for every element with tagname of interest (only):
1. serializes a div (and metadata marking the element's xpath).
2. calculates all text-content for the given element, including descendant elements which are *not* themselves tags of interest.
3. if any such text-content was found, serializes a "headers" (span.headers) along with this text (span.chunk).

to calculate the "headers" of an element:
1. recursively gets the *nearest* prior-siblings for headings of *each* level
2. recursively repeats that step#1 for each ancestor (regardless of tag)
n.b. this recursion is only performed (beginning with) elements which are
both (1) tags-of-interest and (2) have their own text-content.
-->
<xsl:stylesheet version="1.0"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns="http://www.w3.org/1999/xhtml">
	
	<xsl:param name="tags">div|p|blockquote|ol|ul</xsl:param>
	
	<xsl:template match="/">
		<html>
			<head>
				<style>
					div {
						border: solid;
						margin-top: .5em;
						padding-left: .5em;
					}
					
					h1, h2, h3, h4, h5, h6 {
						margin: 0;
					}
					
					.xpath {
						color: blue;
					}
					.chunk {
						margin: .5em 1em;
					}
				</style>
			</head>
			<body>
				<!-- create "filtered tree" with only tags of interest -->
				<xsl:apply-templates select="*" />
			</body>
		</html>
	</xsl:template>
	
	<xsl:template match="*">
		<xsl:choose>
			<!-- tags of interest get serialized into the filtered tree (and recurse down child elements) -->
			<xsl:when test="contains(
				concat('|', $tags, '|'),
				concat('|', local-name(), '|'))">
			
				<xsl:variable name="xpath">
					<xsl:apply-templates mode="xpath" select="." />
				</xsl:variable>
				<xsl:variable name="txt">
					<!-- recurse down child text-nodes and elements -->
					<xsl:apply-templates mode="text" />
				</xsl:variable>
				<xsl:variable name="txt-norm" select="normalize-space($txt)" />
				
				<div title="{$xpath}">
					
					<small class="xpath">
						<xsl:value-of select="$xpath" />
					</small>
					
					<xsl:if test="$txt-norm">
						<xsl:variable name="headers">
							<xsl:apply-templates mode="headingsWithAncestors" select="." />
						</xsl:variable>
						
						<xsl:if test="normalize-space($headers)">
							<span class="headers">
								<xsl:copy-of select="$headers" />
							</span>
						</xsl:if>
					
						<p class="chunk">
							<xsl:value-of select="$txt-norm" />
						</p>
					</xsl:if>
					
					<xsl:apply-templates select="*" />
				</div>
			</xsl:when>
			
			<!-- all other tags get "skipped" and recurse down child elements -->
			<xsl:otherwise>
				<xsl:apply-templates select="*" />
			</xsl:otherwise>
		</xsl:choose>
	</xsl:template>
	
	
	<!-- text mode:
	prints text nodes;
	for elements, recurses down child nodes (text and elements) *except* certain exceptions:
		tags of interest (handled in their own list-mode match),
		non-content text (e.g. script|style)
	-->
	
	<!-- ignore non-content text -->
	<xsl:template mode="text" match="
		script|style" />
	<!-- for all other elements *except tags of interest*, recurse on child-nodes (text and elements) -->
	<xsl:template mode="text" match="*">
		<xsl:choose>
			<!-- ignore tags of interest -->
			<xsl:when test="contains(
				concat('|', $tags, '|'),
				concat('|', local-name(), '|'))" />
			
			<xsl:otherwise>
				<xsl:apply-templates mode="text" />
			</xsl:otherwise>
		</xsl:choose>
	</xsl:template>
	
	
	<!-- xpath mode:
	return an xpath which matches this element uniquely
	-->
	<xsl:template mode="xpath" match="*">
		<!-- recurse up parents -->
		<xsl:apply-templates mode="xpath" select="parent::*" />
		
		<xsl:value-of select="name()" />
		<xsl:text>[</xsl:text>
		<xsl:value-of select="1+count(preceding-sibling::*)" />
		<xsl:text>]/</xsl:text>
	</xsl:template>
	
	
	<!-- headingsWithAncestors mode:
	recurses up parents (ALL ancestors)
	-->
	<xsl:template mode="headingsWithAncestors" match="*">
		<!-- recurse -->
		<xsl:apply-templates mode="headingsWithAncestors" select="parent::*" />
		
		<xsl:apply-templates mode="headingsWithPriorSiblings" select=".">
			<xsl:with-param name="maxHead" select="6" />
		</xsl:apply-templates>
	</xsl:template>
	
	
	<!-- headingsWithPriorSiblings mode:
	recurses up preceding-siblings
	-->
	<xsl:template mode="headingsWithPriorSiblings" match="*">
		<xsl:param name="maxHead" />
		<xsl:variable name="headLevel" select="number(substring(local-name(), 2))" />
		
		<xsl:choose>
			<xsl:when test="'h' = substring(local-name(), 1, 1) and $maxHead >= $headLevel">
				
				<!-- recurse up to prior sibling; max level one less than current -->
				<xsl:apply-templates mode="headingsWithPriorSiblings" select="preceding-sibling::*[1]">
					<xsl:with-param name="maxHead" select="$headLevel - 1" />
				</xsl:apply-templates>
				
				<xsl:apply-templates mode="heading" select="." />
				
			</xsl:when>
			
			<!-- special case for 'header' tag, serialize child-headers -->
			<xsl:when test="self::header">
				<xsl:apply-templates mode="heading" select="h1|h2|h3|h4|h5|h6" />
				<!--
				we choose not to recurse further up prior-siblings in this case,
				but n.b. the 'headingsWithAncestors' template above will still continue recursion.
				-->
			</xsl:when>
			
			<xsl:otherwise>
				<!-- recurse up to prior sibling; no other work on this element -->
				<xsl:apply-templates mode="headingsWithPriorSiblings" select="preceding-sibling::*[1]">
					<xsl:with-param name="maxHead" select="$maxHead" />
				</xsl:apply-templates>
			</xsl:otherwise>
			
		</xsl:choose>
	</xsl:template>
	
	<xsl:template mode="heading" match="h1|h2|h3|h4|h5|h6">
		<xsl:copy>
			<xsl:value-of select="normalize-space(.)" />
		</xsl:copy>
	</xsl:template>
	
</xsl:stylesheet>
