-- EmployeeDB SQL 

-- Q1: Employees per Department
-- Find the number of employees in each department.
SELECT d.dept_name,
       COUNT(de.emp_no) AS emp_count
FROM departments d
JOIN dept_emp de
    ON d.dept_no = de.dept_no
GROUP BY d.dept_name
ORDER BY emp_count DESC;


-- Q2: Highest Salaries
-- List the top 5 employees with the highest salaries.
SELECT e.emp_no,
       e.first_name,
       e.last_name,
       s.salary
FROM employees e
JOIN salaries s
    ON e.emp_no = s.emp_no
ORDER BY s.salary DESC
LIMIT 5;


-- Q3: Current Managers
-- Find the current manager of each department.
SELECT d.dept_name,
       CONCAT(e.first_name, ' ', e.last_name) AS manager_name
FROM departments d
JOIN dept_manager dm
    ON d.dept_no = dm.dept_no
JOIN employees e
    ON dm.emp_no = e.emp_no
WHERE dm.to_date = '9999-01-01'
ORDER BY d.dept_name;


-- Q4: Salary Growth > 50%
-- Hint: Compare MIN(salary) and MAX(salary) for each employee.
SELECT e.emp_no,
       e.first_name,
       e.last_name,
       ROUND(((MAX(s.salary) - MIN(s.salary)) / MIN(s.salary)) * 100, 2) AS growth_percent
FROM employees e
JOIN salaries s
    ON e.emp_no = s.emp_no
GROUP BY e.emp_no, e.first_name, e.last_name
HAVING ((MAX(s.salary) - MIN(s.salary)) / MIN(s.salary)) * 100 > 50
ORDER BY growth_percent DESC;


-- Q5: Longest Serving Employees
SELECT e.emp_no,
       CONCAT(e.first_name, ' ', e.last_name) AS name,
       ROUND(DATEDIFF(CURDATE(), e.hire_date) / 365, 2) AS years_of_service
FROM employees e
ORDER BY years_of_service DESC
LIMIT 5;

