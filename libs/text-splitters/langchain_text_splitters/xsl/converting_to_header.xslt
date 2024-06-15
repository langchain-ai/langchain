<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <!-- Copy all nodes and attributes by default -->
  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>

  <!-- Match any element that has a font-size attribute larger than 20px -->
  <xsl:template match="*[@style[contains(., 'font-size')]]">
    <!-- Extract the font size value from the style attribute -->
    <xsl:variable name="font-size" select="substring-before(substring-after(@style, 'font-size:'), 'px')" />
    <!-- Check if the font size is larger than 20 -->
    <xsl:choose>
      <xsl:when test="$font-size > 20">
        <!-- Replace the element with a header tag -->
        <h1>
          <xsl:apply-templates select="@*|node()"/>
        </h1>
      </xsl:when>
      <xsl:otherwise>
        <!-- Keep the original element -->
        <xsl:copy>
          <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
</xsl:stylesheet>