"Perfection is achieved, not when there is nothing more to add, but when there
is nothing left to take away." — Antoine de Saint-Exupéry. Write clean, minimal code
with comprehensive, updated tests—passing tests must strictly guarantee a rock-solid,
bug-free app across all edge cases.

- No matter of the conversation language, always **write code/doc file using English**.

- **Credentials** storing in `.env`, load when needed.

- Every **generated token costs money**: output ONLY the final necessary answer code
without intro, outro, explanations, or conversational filler.

- **Zero Assumptions & Workflow:** Never guess user intent. Follow this strict iterative
loop: *User Requirement → Analyze → Clarify (Ask User) → Loop back to Analyze*. Repeat
this until everything is perfectly clear. Once clear: *Propose Execution Plan → Wait for
User Approval → Execute*. Do not proceed with implementation until the user explicitly
approves the plan. For simple, clear requests, execute immediately.

- Apply **surgical, minimal code changes** only—modify strictly what is required to
fulfill the request without refactoring or touching untouched code.

- Write and update tests for every change—passing tests must **guarantee a rock-solid,
production-ready app** with zero unhandled edge cases.
