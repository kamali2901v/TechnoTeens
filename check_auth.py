from auth import verify_login

result = verify_login("officer1", "password123")
print("Correct password result:", result)

result_wrong = verify_login("officer1", "wrongpassword")
print("Wrong password result:", result_wrong)