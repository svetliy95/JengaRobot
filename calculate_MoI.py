x = 0.075
y = 0.025
z = 0.015

m = 0.01467

m_x = 1/12 * m * (y**2 + z**2)
m_y = 1/12 * m * (x**2 + z**2)
m_z = 1/12 * m * (y**2 + x**2)

print(m_x / m, m_y / m, m_z / m)
print(m/(x*y*z))