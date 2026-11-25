// Course structure: parts, modules and the planned title of every day.
// Change titles here; a day becomes a full lesson when its notes are in content/notes.

export type Module = {
  n: number;
  slug: string;
  title: string;
  short: string;
  colour: string;
  goal: string;
  learn: string[];
  tools: string[];
  capstone: string;
  days: string[]; // 10 planned titles, in order
};

export type Part = { id: string; title: string; modules: number[] };

export const COURSE = {
  title: "Prescriptive Analytics & Decision Intelligence",
  subtitle: "From a forecast and a set of rules to the best action, and a decision a manager trusts.",
  days: 80,
  hoursPerDay: 5,
};

export const PARTS: Part[] = [
  { id: "I", title: "Optimization", modules: [1, 2, 3] },
  { id: "II", title: "Combinatorial Decisions", modules: [4] },
  { id: "III", title: "Uncertainty", modules: [5, 6] },
  { id: "IV", title: "Trade-offs and Delivery", modules: [7, 8] },
];

export const MODULES: Module[] = [
  {
    n: 1,
    slug: "linear-programming",
    title: "Decision Modelling and Linear Programming",
    short: "Linear Programming",
    colour: "#4f9cf9",
    goal: "Turn a business problem into a paper model and solve it in Pyomo with HiGHS, reading data from DuckDB.",
    learn: [
      "Place a question on the analytics ladder and write any decision as decisions, objective, constraints and data",
      "Check plans, find corners by hand, and solve linear programs with a solver",
      "Read the geometry of an LP: feasible regions, corners, special cases",
      "Build data-driven Pyomo models and debug infeasible or unbounded ones",
      "Plan production and inventory over many weeks, with known-answer tests",
    ],
    tools: ["Python", "DuckDB", "pandas", "SciPy (HiGHS)", "Pyomo", "matplotlib", "pytest"],
    capstone: "A 12-week production and inventory plan for the private-label bakery, tested and explained.",
    days: [
      "From Prediction to Prescription",
      "Spreadsheet First: The Product Mix",
      "The Geometry of Linear Programs",
      "Your First Pyomo Model",
      "Mini-project: Blending and Meal Kits",
      "Data-Driven Models from DuckDB",
      "When Models Break: Infeasible and Unbounded",
      "Multi-Period Production and Inventory",
      "Units, Scaling and Known-Answer Tests",
      "Capstone: The 12-Week Bakery Plan",
    ],
  },
  {
    n: 2,
    slug: "sensitivity-networks",
    title: "Sensitivity, Duality and Network Flows",
    short: "Sensitivity & Networks",
    colour: "#3ecf8e",
    goal: "A distribution plan from 3 DCs to 40 stores, with a what-if memo built on shadow prices.",
    learn: [
      "Read shadow prices, reduced costs and ranges, and turn them into decisions",
      "Model transportation, assignment, min-cost flow, shortest path and max flow",
      "Feed forecasts into a network model and compare plan with actual",
    ],
    tools: ["Pyomo", "HiGHS", "DuckDB", "NetworkX"],
    capstone: "A weekly distribution plan and a one-page shadow-price memo: which capacity should we buy?",
    days: [
      "Shadow Prices and Duality",
      "Reduced Costs and Ranging",
      "What-If Analysis at Scale",
      "Transportation and Assignment",
      "Mini-project: Assigning Stores to DCs",
      "Min-Cost Flow and Cross-Docks",
      "Shortest Paths and Bottlenecks",
      "Multi-Commodity and Multi-Period Flows",
      "Feeding the Model with Forecasts",
      "Capstone: The Weekly Distribution Plan",
    ],
  },
  {
    n: 3,
    slug: "integer-optimization",
    title: "Integer and Combinatorial Optimization",
    short: "Integer Optimization",
    colour: "#f5a524",
    goal: "Model yes/no decisions correctly, and know why a MIP is slow and how to speed it up.",
    learn: [
      "Turn business rules into binary logic, fixed costs and big-M constraints",
      "Solve facility location, knapsack, bin packing and cutting stock",
      "Read MIP solver logs and strengthen weak formulations",
    ],
    tools: ["Pyomo", "HiGHS", "Gurobi (optional)"],
    capstone: "A DC network redesign (open, close, resize) with a 5-year cost case and a gap report.",
    days: [
      "Binary Variables and Business Logic",
      "Fixed Costs and Big-M",
      "Facility Location",
      "Knapsack, Bin Packing and Cutting Stock",
      "Mini-project: Which Cross-Docks to Open",
      "How MIP Solvers Work",
      "Strong Formulations and Symmetry",
      "Making MIPs Fast: HiGHS vs Gurobi",
      "Piecewise-Linear Costs and Discounts",
      "Capstone: Redesigning the DC Network",
    ],
  },
  {
    n: 4,
    slug: "scheduling-routing",
    title: "Scheduling, Routing and Constraint Programming",
    short: "Scheduling & Routing",
    colour: "#e5484d",
    goal: "A legal weekly staff roster and daily delivery routes, built with OR-Tools.",
    learn: [
      "Choose between constraint programming and MIP",
      "Build rosters that respect labour rules, preferences and fairness",
      "Route a mixed fleet with time windows, and know when heuristics win",
    ],
    tools: ["OR-Tools CP-SAT", "OR-Tools routing", "Python"],
    capstone: "Next week's roster and daily routes, compared with the current plan on kilometres, overtime and late deliveries.",
    days: [
      "Constraint Programming with CP-SAT",
      "Staff Scheduling I: Coverage and Labour Rules",
      "Staff Scheduling II: Preferences and Fairness",
      "Job-Shop Scheduling on the Bakery Lines",
      "Mini-project: A 4-Week Warehouse Roster",
      "Vehicle Routing I: TSP and CVRP",
      "Vehicle Routing II: Time Windows and Mixed Fleets",
      "Heuristics and Metaheuristics",
      "Decomposition and Rolling Horizons",
      "Capstone: Next Week's Roster and Routes",
    ],
  },
  {
    n: 5,
    slug: "simulation",
    title: "Simulation and Monte Carlo",
    short: "Simulation",
    colour: "#a78bfa",
    goal: "Stress-test any plan against uncertain demand, travel times and supply, and show its risk, not just its average.",
    learn: [
      "Fit input distributions and treat forecast errors as inputs",
      "Run Monte Carlo and discrete-event simulations, checked against queueing formulas",
      "Design simulation experiments and search designs with a simulator in the loop",
    ],
    tools: ["NumPy", "SciPy", "SimPy"],
    capstone: "A stress test of the Module 3 network and Module 4 routes, with a risk table for executives.",
    days: [
      "The Flaw of Averages",
      "Input Distributions and Forecast Errors",
      "Monte Carlo of a Plan",
      "Correlated Risks and Disruptions",
      "Mini-project: How Often Do We Run Out?",
      "Discrete-Event Simulation with SimPy",
      "Queueing Theory as a Check",
      "Designing Simulation Experiments",
      "Simulation-Optimization",
      "Capstone: Stress-Testing the Network",
    ],
  },
  {
    n: 6,
    slug: "stochastic-optimization",
    title: "Optimization under Uncertainty",
    short: "Under Uncertainty",
    colour: "#22b8cf",
    goal: "A replenishment policy that holds up when the forecast is wrong, with its value proven.",
    learn: [
      "Set order quantities and safety stock from the cost of too much and too little",
      "Build two-stage stochastic, chance-constrained and robust models",
      "Add risk to the objective (CVaR) and find optimal policies with dynamic programming",
    ],
    tools: ["Pyomo", "HiGHS", "Gurobi (optional)", "NumPy"],
    capstone: "Deterministic vs stochastic vs robust replenishment, simulated on held-out weeks, with a recommendation.",
    days: [
      "The Newsvendor",
      "Inventory Policies and Safety Stock",
      "Two-Stage Stochastic Programming",
      "Sample Average Approximation",
      "Mini-project: VSS and EVPI",
      "Chance Constraints and Service Levels",
      "Robust Optimization",
      "Risk Measures: CVaR",
      "MDPs, Dynamic Programming and RL",
      "Capstone: A Replenishment Policy That Holds Up",
    ],
  },
  {
    n: 7,
    slug: "decision-analysis",
    title: "Decision Analysis and Trade-offs",
    short: "Decision Analysis",
    colour: "#f76b15",
    goal: "An executive can pick a point on a cost–carbon–service frontier, knowing what each choice gives up.",
    learn: [
      "Structure decisions with trees, utility and the value of information",
      "Trace Pareto frontiers with the epsilon-constraint method",
      "Optimize prices and markdowns, and communicate trade-offs honestly",
    ],
    tools: ["Pyomo", "HiGHS", "matplotlib"],
    capstone: "A board-ready trade-off memo for the 2030 carbon target, with three options and their costs.",
    days: [
      "Decision Trees and Expected Value",
      "Risk Attitude and Utility",
      "The Value of Information",
      "Bayesian Decision Analysis",
      "Mini-project: Piloting Electric Trucks",
      "Multi-Objective Optimization",
      "Pareto Frontiers: Cost, Carbon, Service",
      "Pricing and Markdown Optimization",
      "Communicating Trade-offs",
      "Capstone: The 2030 Carbon Memo",
    ],
  },
  {
    n: 8,
    slug: "delivery",
    title: "Decision Intelligence Delivery",
    short: "Delivery",
    colour: "#e93d82",
    goal: "An optimization model running as a tested service that re-plans when data change and explains its plans.",
    learn: [
      "Test optimization models and explain plans in plain language",
      "Serve a model through FastAPI and a Streamlit scenario app",
      "Re-optimize, monitor and govern decisions in production",
    ],
    tools: ["FastAPI", "Streamlit", "pytest", "GitHub Actions", "Docker"],
    capstone: "The decision simulation: re-plan after a shock and defend it to an executive panel.",
    days: [
      "From Notebook to Product",
      "Testing Optimization Models",
      "Explaining a Plan",
      "A Decision Service with FastAPI",
      "Mini-project: A Streamlit Scenario App",
      "Re-optimization and Plan Stability",
      "Monitoring Plans and Models",
      "Human in the Loop and Governance",
      "Production Readiness",
      "Final Capstone: The Decision Simulation",
    ],
  },
];

export function moduleBySlug(slug: string) {
  return MODULES.find((m) => m.slug === slug);
}

export function moduleOfDay(day: number) {
  return MODULES[Math.floor((day - 1) / 10)];
}
