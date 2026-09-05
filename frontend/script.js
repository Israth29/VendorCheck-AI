document.getElementById("loginForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  const employeeId = document.getElementById("employeeId").value;
  const employeeName = document.getElementById("employeeName").value;
  const companyCode = document.getElementById("companyCode").value;
  const errorMsg = document.getElementById("errorMsg");

  try {
    const response = await fetch("http://127.0.0.1:8000/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        employee_id: employeeId,
        name: employeeName,
        company_code: companyCode
      })
    });

    const data = await response.json();

    if (data.success) {
      localStorage.setItem("userName", data.name);
      localStorage.setItem("userRole", data.role);
      localStorage.setItem("userEmployeeId", employeeId);

      if (data.role === "hr") {
        window.location.href = "hr-dashboard.html";
      } else if (data.role === "procurement") {
        window.location.href = "procurement-dashboard.html";
      }
    } else {
      errorMsg.style.color = "#f87171";
      errorMsg.innerText = data.message;
    }
  } catch (err) {
    errorMsg.innerText = "Cannot connect to server.";
  }
});