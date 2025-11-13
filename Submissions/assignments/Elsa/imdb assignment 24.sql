-- IMDb SQL Starter File

-- Q1: Movies by Director
SELECT d.name AS director_name,
       COUNT(m.movie_id) AS movie_count
FROM directors d
JOIN movie_directors md 
    ON d.director_id = md.director_id
JOIN movies m 
    ON md.movie_id = m.movie_id
GROUP BY d.name
ORDER BY movie_count DESC;

-- Q2: Popular Genres
--List genres where the average rating is above 7.5.
SELECT genre,
       ROUND(AVG(rating), 2) AS avg_rating
FROM movies
GROUP BY genre
HAVING AVG(rating) > 7.5
ORDER BY avg_rating DESC;


-- Q3: Actor with Most Movies
SELECT a.name AS actor_name,
       COUNT(ma.movie_id) AS movie_count
FROM actors a
JOIN movie_actors ma 
    ON a.actor_id = ma.actor_id
GROUP BY a.name
ORDER BY movie_count DESC
LIMIT 1;


-- Q4: Directors with 3 Consecutive Years
-- Hint: Use window functions (LAG, LEAD) or self-joins.
SELECT DISTINCT d.name AS director_name
FROM directors d
JOIN movie_directors md 
    ON d.director_id = md.director_id
JOIN movies m 
    ON md.movie_id = m.movie_id
WHERE EXISTS (
    SELECT 1
    FROM movies m1
    JOIN movie_directors md1 ON m1.movie_id = md1.movie_id
    WHERE md1.director_id = d.director_id
    AND EXISTS (
        SELECT 1
        FROM movies m2
        JOIN movie_directors md2 ON m2.movie_id = md2.movie_id
        WHERE md2.director_id = d.director_id
        AND (m2.year = m1.year + 1)
        AND EXISTS (
            SELECT 1
            FROM movies m3
            JOIN movie_directors md3 ON m3.movie_id = md3.movie_id
            WHERE md3.director_id = d.director_id
            AND (m3.year = m1.year + 2)
        )
    )
);


-- Q5: Actor-Director Collaborations
SELECT 
    a.name AS actor_name,
    d.name AS director_name,
    COUNT(*) AS collaboration_count
FROM Movie_Crew AS ac
JOIN Movie_Crew AS dc 
    ON ac.movie_id = dc.movie_id
    AND ac.role = 'actor'
    AND dc.role = 'director'
JOIN People AS a 
    ON ac.person_id = a.person_id
JOIN People AS d 
    ON dc.person_id = d.person_id
GROUP BY a.name, d.name
HAVING COUNT(*) >= 3
ORDER BY collaboration_count DESC;


