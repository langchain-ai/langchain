-- Import client's _sales file

-- Total records
select count(id) FROM nissan_canada_10003_pathfinder_45_sales;

-- Unique households
select count(t1.Street1) from
(SELECT DISTINCT s.Street1,s.`Postal Code` FROM toolbox.nissan_canada_10003_pathfinder_45_sales s) as t1;

-- Create hashed values so they can be matched to _mail file

-- UPDATE nissan_canada_10003_pathfinder_45_sales set hash_firstname=upper(sha2(`First Name`,256)),hash_lastname=upper(sha2(`Last Name`,256)),hash_street1=upper(SHA2(Street1,256)),hash_street2=upper(SHA2(Street2,256)),hash_city=upper(SHA2(City,256)),hash_province=upper(SHA2(Province,256)),hash_postalcode=upper(sha2(`Postal Code`,256));



SELECT distinct s.* FROM nissan_canada_10003_pathfinder_45_sales s
inner JOIN nissan_canada_10003_pathfinder_mail m ON s.hash_street1=m.Street1 AND s.hash_postalcode=m.`Postal Code`;


SELECT distinct s.* FROM nissan_canada_10003_pathfinder_45_sales s
inner JOIN nissan_canada_10003_pathfinder_mail m ON s.hash_street1=m.Street1;

SELECT DISTINCT s1.`Customer ID`,s1.Street1,s1.`Postal Code`,s2.`Customer ID`,s2.Street1,s2.`Postal Code` FROM nissan_canada_10003_pathfinder_45_sales s1 INNER JOIN nissan_canada_10001_rogue_45_sales s2 ON s1.`Customer ID`=s2.`Customer ID`;


SELECT * FROM nissan_canada_10003_pathfinder_mail WHERE City=sha2('TORONTO',256);


/*
UPDATE nissan_canada_10003_pathfinder_45_sales s
inner JOIN nissan_canada_10003_pathfinder_mail m ON s.hash_street1=m.Street1 AND s.hash_postalcode=m.`Postal Code`
set s.`Match`=true;
*/

select * FROM nissan_canada_10003_pathfinder_45_sales WHERE `Match`;

SELECT Province,COUNT(id) FROM nissan_canada_10003_pathfinder_45_sales WHERE `Match` GROUP BY Province ORDER BY Province;
