# RAF ICD Backend - Build Instructions

## Fixed: Lombok Compilation Errors

The compilation errors were due to Lombok not being properly configured in Maven.

### What Was Fixed:
1. Added Maven Compiler Plugin with Lombok annotation processor
2. Specified explicit Lombok version (1.18.30)
3. Changed Lombok scope to `provided`

---

## How to Build Now

### Step 1: Clean Previous Build
```powershell
cd C:\Users\Coditas\Desktop\Projects\RAFgenAI\backend
mvn clean
```

### Step 2: Build Project
```powershell
mvn install -DskipTests
```

This will:
- Download Lombok if needed
- Process Lombok annotations
- Generate getters, setters, constructors
- Compile all Java files
- Package the application

### Step 3: Run Application
```powershell
mvn spring-boot:run
```

---

## If You Still Get Errors

### Alternative: Build Without Lombok (Manual DTOs)

If Lombok still doesn't work, we can remove Lombok and add manual getters/setters.

Let me know if you need this fallback option.

---

## Expected Output

When build succeeds, you should see:
```
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  XX.XXX s
[INFO] Finished at: 2026-02-17T...
[INFO] ------------------------------------------------------------------------
```

When running:
```
Started IcdBackendApplication in X.XXX seconds
```

---

## Quick Commands

```powershell
# Clean and build
mvn clean install -DskipTests

# Run
mvn spring-boot:run

# Or both in one command
mvn clean install -DskipTests && mvn spring-boot:run
```
