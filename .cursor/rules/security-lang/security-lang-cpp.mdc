---
description: 
globs: **/*.cpp,**/*.cc,**/*.cxx,**/*.hpp,**/*.hh,**/*.hxx,**/*.h++,**/*.C
alwaysApply: false
---
# Secure C++ Development

These rules apply to all C++ source and header files in the repository and aim to prevent memory corruption, resource leaks, undefined behavior, and security vulnerabilities through modern C++ best practices.

All violations must include a clear explanation of which rule was triggered and why, to help developers understand and fix the issue effectively.  
Generated code must not violate these rules. If a rule is violated, a comment must be added explaining the issue and suggesting a correction.

## 1. Prevent Use After Free and Double Free (CWE-416, CWE-415)
- **Rule:** Use `std::unique_ptr`, `std::shared_ptr`, or `std::weak_ptr` instead of raw `new`/`delete`. Never access memory after it has been freed. Do not free the same memory twice. Smart pointers and RAII patterns automatically prevent use-after-free and double-free vulnerabilities.

## 2. Avoid Raw Pointers for Ownership
- **Rule:** Do not use raw pointers (`T*`) to express ownership. Use smart pointers or containers. Raw pointers should only be used for non-owning references. Always verify non-owning pointers are not null before dereferencing.

## 3. Initialize All Variables (CWE-824)
- **Rule:** Always initialize variables, including pointers, at declaration. Use uniform initialization `{}` to prevent narrowing conversions. Never leave objects in an indeterminate state. Accessing uninitialized pointers or variables leads to undefined behavior and potential security vulnerabilities.

## 4. Prevent Use-After-Move
- **Rule:** After moving from an object with `std::move()`, do not access its value unless it has been reassigned. Document moved-from states clearly.

## 5. Prevent Out-of-Bounds Write and Read (CWE-787, CWE-125, CWE-119, CWE-676)
- **Rule:** Use STL containers (`std::vector`, `std::array`, `std::string`, `std::span` for views) with bounds-checked access (`.at()`) instead of C-style arrays and manual index looping. When converting legacy C-style arrays, use `std::to_array` (C++20) to preserve size information and enable bounds-checked access. Never use potentially dangerous functions like `strcpy`, `sprintf`, `gets`, `scanf`, or other unbounded string functions. Prefer standard algorithms with `.at()` over unchecked `operator[]` in security-critical code. Out-of-bounds memory access is a leading cause of exploitable vulnerabilities.

## 6. Check Iterator Validity
- **Rule:** Verify iterators remain valid after container modifications. Operations like `insert()`, `erase()`, or reallocation may invalidate iterators, causing undefined behavior.

## 7. Use `std::string` and `std::string_view` Safely
- **Rule:** Use `std::string` for ownership and `std::string_view` for non-owning references, ensuring the underlying data outlives the view. Avoid mixing std::string with C-style string manipulation.

## 8. Handle Exceptions Safely
- **Rule:** Destructors must be `noexcept` and should never throw. Use exception-safe coding (RAII, strong exception guarantee) to prevent resource leaks during stack unwinding.

## 9. Avoid Uninitialized or Dangling References
- **Rule:** Do not return references or pointers to local variables. Ensure reference parameters outlive their usage. Use `std::optional` or `std::reference_wrapper` when appropriate.

## 10. Prefer `const` and `constexpr`
- **Rule:** Mark variables, references, pointers, and member functions `const` whenever possible. Use `constexpr` for compile-time constants to prevent accidental modification.

## 11. Avoid Implicit Type Conversions (CWE-704)
- **Rule:** Use `explicit` constructors for single-argument constructors to prevent unintended implicit conversions. Avoid C-style casts; prefer `static_cast`, `dynamic_cast`, `const_cast`, or `reinterpret_cast` with documented justification. Incorrect type conversions can lead to security vulnerabilities including integer truncation, sign errors, and buffer overflows.

## 12. Validate Array and Container Access (CWE-787, CWE-125)
- **Rule:** Always validate indices and sizes before accessing arrays or containers. Use `.at()` for bounds-checked access or check with `.size()` before using `operator[]`. Never assume array bounds without explicit validation. Out-of-bounds reads and writes are actively exploited vulnerabilities.

## 13. Prevent NULL Pointer Dereference (CWE-476)
- **Rule:** Prefer `std::optional` for values that may be absent, or `std::unique_ptr`/`std::shared_ptr` for owned resources—these types make null states explicit and compiler-checkable. When raw pointers are unavoidable, perform explicit null checks before dereferencing. Dereferencing null pointers leads to crashes and denial of service.

## 14. Prevent Integer Overflow and Wraparound (CWE-190, CWE-195, CWE-196, CWE-197)
- **Rule:** Use safe integer libraries (e.g., SafeInt, Boost.SafeNumerics) for all arithmetic that computes array indices, allocation sizes, or loop bounds. Both signed overflow (undefined behavior) and unsigned wraparound (defined but dangerous) can lead to buffer overflows and memory corruption. Safe integer types detect overflow at runtime and fail safely instead of silently producing incorrect values. Additionally, validate integer type conversions—truncation (e.g., `int64_t` to `int32_t`) and sign conversion (e.g., signed to unsigned) can introduce vulnerabilities even without arithmetic overflow.

## 15. Avoid Undefined Behavior
- **Rule:** Do not invoke undefined behavior. Use compiler flags (`-fsanitize=undefined`, `-fsanitize=address`) to detect violations in development and testing builds, but not for standard release builds due to performance overhead. Enable sanitizers during CI/CD and local development to catch issues early.

## 16. Use Thread-Safe Constructs
- **Rule:** Protect shared mutable state with `std::mutex`, `std::lock_guard`, `std::unique_lock`, or `std::scoped_lock`. Prefer `std::jthread` (C++20) over `std::thread` for automatic joining. Avoid data races. Use `std::atomic` for lock-free operations when appropriate.

## 17. Avoid `reinterpret_cast` and Type Punning
- **Rule:** Do not use `reinterpret_cast` or type punning (e.g., via unions or pointer casts) except in well-justified low-level code with thorough documentation and testing.

## 18. Validate Template Parameters
- **Rule:** Use concepts (C++20) or `static_assert` to constrain and validate template parameters. Ensure template instantiations do not introduce security vulnerabilities or undefined behavior.

## 19. Prevent Virtual Function Vulnerabilities
- **Rule:** Mark final classes and functions as `final`. Avoid calling virtual functions in constructors or destructors where polymorphism does not apply and can cause undefined behavior.

## 20. Use Modern Algorithms and Ranges
- **Rule:** Prefer standard algorithms (`<algorithm>`, `<numeric>`) and ranges (C++20) over manual index loops to reduce off-by-one errors and improve correctness. Range-based algorithms eliminate index arithmetic that can lead to security vulnerabilities.

## 21. Avoid `goto` and Macros
- **Rule:** Do not use `goto` statements. Avoid preprocessor macros for constants and functions; use `constexpr`, `inline`, or templates instead.

## 22. Handle User Input Safely
- **Rule:** Never trust user input. Validate all external input (files, network, command-line arguments) for type, length, format, and range before using in operations like memory allocation, array access, or logic branches.

## 23. Prevent Command Injection
- **Rule:** Avoid `system()`, `popen()`, or shell execution with user-controlled input. Use direct OS APIs (e.g., `std::filesystem`, POSIX `exec` family) or sanitize input rigorously.

## 24. Secure Cryptographic Operations
- **Rule:** Use established cryptographic libraries (OpenSSL, Botan, libsodium). Never implement custom cryptography. Ensure proper key management, random number generation, and secure destruction of sensitive data.

## 25. Zero Sensitive Memory
- **Rule:** Explicitly zero out memory containing sensitive data (passwords, keys, tokens) using `std::fill`, `memset_s`, or secure_zero functions before deallocation to prevent information leakage.

## 26. Use Compiler Security Features
- **Rule:** Enable compiler security flags: `-fstack-protector-strong`, `-D_FORTIFY_SOURCE=2`, `-fPIE`, `-pie`, `-Wl,-z,relro,-z,now` (on Linux/GCC/Clang) to enable stack protection, position-independent code, and full RELRO.

## 27. Avoid Multiple Inheritance and Diamond Problems
- **Rule:** Use multiple inheritance cautiously. Prefer composition over inheritance. Use virtual inheritance sparingly and only when necessary to avoid ambiguity and complexity.

## 28. Validate Lifetime Dependencies
- **Rule:** Ensure objects with dependencies (e.g., observer pointers, callbacks, lambdas capturing references) do not outlive the objects they reference. Use weak pointers or lifetime managers where appropriate.

## 29. Secure Third-Party Libraries
- **Rule:** Use well-maintained, trusted libraries. Regularly update dependencies and audit for known vulnerabilities (e.g., CVE databases). Prefer header-only or statically linked libraries with security track records.

## 30. Do Not Mix C and C++ Memory Management (CWE-762)
- **Rule:** Do not mix `malloc`/`free` with `new`/`delete`, or with smart pointers. Mismatched memory management routines lead to undefined behavior and memory corruption. Use consistent memory management idioms throughout the codebase.

## 31. Use Static Analysis and Sanitizers
- **Rule:** Integrate static analysis tools (`clang-tidy`, `cppcheck`, `PVS-Studio`) and runtime sanitizers into CI/CD pipelines to catch security issues early. Create separate build configurations for each sanitizer: `-fsanitize=address` (memory errors), `-fsanitize=thread` (data races), `-fsanitize=undefined` (UB detection), and `-fsanitize=null` (null pointer dereference). For detecting uninitialized memory reads (CWE-457), consider `-fsanitize=memory` in isolated environments where full instrumentation of all dependencies (including libc++) is feasible. Run sanitizer builds in development and testing only—do not enable sanitizers in release builds due to performance overhead.

## 32. Prevent Time-of-Check-Time-of-Use (TOCTOU) Errors
- **Rule:** Avoid race conditions between checking a resource's state and using it. Use atomic operations or open files with appropriate flags (`O_CREAT | O_EXCL`) instead of checking existence then opening.

## 33. Secure Random Number Generation
- **Rule:** Use cryptographically secure random number generators (`std::random_device` with appropriate entropy source, or platform-specific APIs like `/dev/urandom`, `CryptGenRandom`, `arc4random`) for security-sensitive operations. Do not use `rand()` or `std::mt19937` for security purposes.

## 34. Handle Signal Handlers Safely
- **Rule:** Signal handlers must only call async-signal-safe functions. Avoid accessing shared state without proper synchronization. Prefer modern alternatives like `signalfd` (Linux) or avoid signals entirely when possible.

## 35. Secure File Operations
- **Rule:** When creating files, use restrictive permissions (e.g., `0600` for sensitive files). Avoid creating predictable temporary filenames; use `std::filesystem::temp_directory_path()` with unique random names or `mkstemp()`. Check return values from all file operations.

## 36. Prevent Format String Vulnerabilities
- **Rule:** Never pass user-controlled strings directly as format arguments to `printf`, `fprintf`, `sprintf`, `syslog`, or similar functions. Use type-safe alternatives like `std::format` (C++20), `fmt` library, or streaming operators.

## 37. Validate Dynamic Casts
- **Rule:** Check the result of `dynamic_cast` for null pointers before dereferencing. Prefer avoiding downcasts when possible by using proper polymorphic design.

## 38. Secure Object Copying
- **Rule:** Follow the Rule of Zero (no explicit destructor/copy/move) or Rule of Five (implement all). Prevent shallow copies of resources. Use `= delete` to explicitly disable copying when objects should not be copied.

## 39. Properly Scope Lock Acquisition
- **Rule:** Always use RAII lock guards (`std::lock_guard`, `std::unique_lock`, `std::scoped_lock`) instead of manual `lock()`/`unlock()` calls. Acquire locks in consistent order across threads to prevent deadlocks.

## 40. Use `std::span` for Array Parameters
- **Rule:** Pass arrays and contiguous sequences as `std::span<T>` (C++20) instead of pointer-and-size pairs or raw arrays. This provides bounds information and enables safer access patterns that remain valid as code evolves.

## 41. Prefer `std::expected` for Recoverable Errors
- **Rule:** Use `std::expected<T, E>` (C++23) or similar types (e.g., `tl::expected`, `boost::outcome`) for error handling instead of exceptions or error codes when appropriate. This makes error paths explicit and analyzable at compile time.

## 42. Leverage Compile-Time Evaluation
- **Rule:** Use `constexpr` and `consteval` (C++20) functions to move validation and computations to compile time when possible. Compile-time errors are always preferable to runtime vulnerabilities.

## 43. Adopt Modern C++ Standards
- **Rule:** Target the latest stable C++ standard your toolchain supports (C++20 or later). Modern features like concepts, ranges, `std::span`, modules, and coroutines provide safer abstractions that will remain relevant in future standards.

## 44. Use Designated Initializers for Clarity
- **Rule:** Use designated initializers (C++20) for aggregate initialization to make field initialization explicit and prevent initialization order errors that could lead to security issues.

## 45. Prefer Modules Over Headers
- **Rule:** When using C++20 or later, prefer modules over traditional headers to reduce macro-related vulnerabilities, improve encapsulation, and prevent unintended name collisions that could lead to security bugs.

## 46. Leverage `std::source_location` for Debugging
- **Rule:** Use `std::source_location` (C++20) instead of preprocessor macros (`__FILE__`, `__LINE__`) for logging and debugging. This provides type-safe, future-proof context information without macro pitfalls.

## 47. Use `std::bit_cast` for Type Reinterpretation
- **Rule:** Replace `reinterpret_cast` and `memcpy`-based type punning with `std::bit_cast` (C++20) when safe type reinterpretation is needed. This provides well-defined behavior and better compiler optimization opportunities.

## 48. Avoid Operator Precedence Errors (CWE-783)
- **Rule:** Use explicit parentheses in complex expressions to make operator precedence clear. Avoid relying on implicit precedence rules that can lead to logic errors, especially in security-critical conditions (authentication, authorization, bounds checks). Use parentheses even when not strictly required for clarity.

## 49. Prevent Type Confusion (CWE-843)
- **Rule:** Avoid accessing resources using incompatible types. Use `std::variant` or `std::any` with proper type checking instead of unions or type punning. Validate types before casting. Type confusion vulnerabilities can lead to arbitrary code execution.

## 50. Encapsulate Critical Data (CWE-766, CWE-767)
- **Rule:** Declare security-critical data members as `private` and provide controlled access through `const` member functions. Avoid public member variables. Use getters that return `const` references or copies to prevent unauthorized modification. Follow the principle of least privilege for data access.

## 51. Apply Security Standards Systematically
- **Rule:** Follow established guidelines (ISO C++ Core Guidelines, SEI CERT C++ Coding Standard, CWE-659, 2024 CWE Top 25) and keep code aligned with evolving security best practices. Regularly review code against updated standards and migrate to safer modern alternatives as they become available. Use white-box fuzz testing to detect memory corruption and security vulnerabilities.