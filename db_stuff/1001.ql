select count(id) FROM nissan_canada_10001_rogue_45_sales;

select count(t1.`First Name`) from
(SELECT DISTINCT s.`First Name`,s.`Last Name`,s.Street1,s.Street2,s.City,s.Province,s.`Postal Code` FROM toolbox.nissan_canada_10001_rogue_45_sales s) as t1;

select count(t1.Street1) from
(SELECT DISTINCT s.Street1,s.`Postal Code` FROM toolbox.nissan_canada_10001_rogue_45_sales s) as t1;

select * FROM toolbox.nissan_canada_10001_rogue_45_sales WHERE
(upper(Street1) IN ('ERROR','LOYALTY','CONQUEST',''))
OR City='' or Province='' or `Postal Code`='';

-- UPDATE nissan_canada_10001_rogue_45_sales set hash_firstname=upper(md5(`First Name`)),hash_lastname=upper(md5(`Last Name`)),hash_street1=upper(md5(Street1)),hash_street2=upper(md5(Street2)),hash_city=upper(md5(City)),hash_province=upper(md5(Province)),hash_postalcode=upper(md5(`Postal Code`));
SELECT distinct s.* FROM nissan_canada_10001_rogue_45_sales s
inner JOIN nissan_canada_10001_rogue_mail m ON s.hash_street1=m.Street1 AND s.hash_postalcode=m.`Postal Code`;
/*
UPDATE nissan_canada_10001_rogue_45_sales s
inner JOIN nissan_canada_10001_rogue_mail m ON s.hash_street1=m.Street1 AND s.hash_postalcode=m.`Postal Code`
set s.`Match`=true;
*/

select * FROM nissan_canada_10001_rogue_45_sales WHERE `Match`;

SELECT Province,COUNT(id) FROM nissan_canada_10001_rogue_45_sales WHERE `Match` GROUP BY Province ORDER BY Province;


SELECT p.Province,COUNT(m.id) FROM nissan_canada_10001_rogue_mail m
inner join
(SELECT 'AB' AS Province,MD5('AB') AS Hash
UNION ALL SELECT 'BC',MD5('BC')
UNION ALL SELECT 'SK',MD5('SK')
UNION ALL SELECT 'MB',MD5('MB')
UNION ALL SELECT 'ON',MD5('ON')
UNION ALL SELECT 'QC',MD5('QC')
UNION ALL SELECT 'NB',MD5('NB')
UNION ALL SELECT 'NS',MD5('NS')
UNION ALL SELECT 'PE',MD5('PE')
UNION ALL SELECT 'NL',MD5('NL')
UNION ALL SELECT 'YT',MD5('YT')
UNION ALL SELECT 'NU',MD5('NU')
UNION ALL SELECT 'NT',MD5('NT')) AS p ON UPPER(p.Hash)=UPPER(m.Province)
 GROUP BY p.Province ORDER BY p.Province;
