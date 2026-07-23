---
layout: default
title: Projects
permalink: /projects/
---

<article class="post projects">
    <header>
        <h1>Projects</h1>
        <p class="tagline">A decade of building tools for people who build things.</p>
    </header>

    <h2 id="work">Work</h2>

    <div class="project">
        <h3><a href="https://www.kope.ai/">KOPE</a> <span class="cv-year">2022 – 2026</span></h3>
        <p class="project-kicker">Configurator platform for industrialized construction</p>
        <img src="{{ '/assets/images/projects/kope.jpg' | relative_url }}" alt="The KOPE configurator: a building model in the 3D viewer with product applications and the datatable open">
        <p>KOPE is a multiplayer B2B SaaS platform for offsite construction, built around a highly performant Three.js viewer optimised for large, detailed building models.</p>
        <p>At its heart, it lets users apply a library of configurable construction products to their building designs, resolving them to a manufacturing level of detail within minutes — each product carrying the logic, options and limitations of its manufacturer.</p>
        <p>I joined when it was marketed as a generic workflow automation tool and pitched the reframing we never looked back from: a place for the industry to simply <em>apply products to their projects</em>.</p>
        <p>Since then I've led the platform team and owned the vision, roadmap and design of the flagship app — entirely no-code product build-up, a flexible datatable for interrogating and visualising data, side-by-side design option comparison, multi-objective evolutionary optimisation, and most recently a suite of 3D model editing tools.</p>
    </div>

    <div class="project">
        <h3><a href="https://market.kope.ai/">KOPE Market</a> <span class="cv-year">2021 – 2026</span></h3>
        <p class="project-kicker">International marketplace for offsite construction</p>
        <img src="{{ '/assets/images/projects/kope-market.jpg' | relative_url }}" alt="A manufacturer profile page on KOPE Market showing products, case studies and locations">
        <p>The marketplace began as a research task: within a month I'd amassed an enormous database of interconnected datapoints — people, places, media, manufacturers, exemplar projects and software.</p>
        <p>We launched it as a standalone, open and free website, bringing transparency to construction's infamously opaque supply chains while acting as our startup's advertising engine.</p>
        <p>I led its development from day one: designing the responsive site, writing all requirements, even specifying the database tables. Today it's fully internationalised and white-labelled — including a version delivered to the NHS New Hospitals Programme — with thousands of products searchable across international certification standards.</p>
    </div>

    <div class="project">
        <h3><a href="https://dynamobim.org/dynamo-core-2-13-release-part-1-3/">Dynamo UI Rebuild</a> <span class="cv-year">2021</span></h3>
        <p class="project-kicker">Contract UI rebuild of visual programming software, for Autodesk</p>
        <img src="{{ '/assets/images/projects/dynamo-rebuild.jpg' | relative_url }}" alt="The rebuilt Dynamo node interface showing code blocks, node connections and context menus">
        <p>A full-circle moment: I first learned to program in Dynamo's interface, and much of this contract's initial workload came from feature requests I had logged on GitHub years earlier.</p>
        <p>Working with Autodesk's product and UI team, I rebuilt the core app UI — the highly configurable node UI, the node library and search, much of the package manager, and a new hierarchical warning system — in XAML, C# and WPF's MVVM architecture, with every build subjected to over 10,000 tests.</p>
        <p>The richer UI initially brought a performance hit ("Dynamo is pushing the reasonable limits of what WPF can achieve"), so I profiled it with dotMemory until the rebuilt app outperformed the old one. Autodesk offered me a job afterwards — but my heart was set on product management.</p>
    </div>

    <div class="project">
        <h3><a href="https://www.researchgate.net/publication/344684613_An_Introspective_Approach_to_Apartment_Design">Homegrown</a> <span class="cv-year">2019 – 2020</span></h3>
        <p class="project-kicker">Machine learning apartment recommendation</p>
        <img src="{{ '/assets/images/projects/homegrown.jpg' | relative_url }}" alt="Homegrown's suggested designs dialog in Revit, ranking previously built apartment layouts by match percentage">
        <p>Automated space planning has been discussed for decades — physics-based systems, Bayesian analysis, evolutionary algorithms, deep learning — yet architects were still designing apartments manually. Homegrown took a novel approach: instead of generic spatial intelligence, work backwards from the firm's own library of built apartment designs.</p>
        <p>The result is a recommendation engine that recognises similarity between spaces regardless of size, shape, mirroring or rotation, delivered as a Revit plugin with zero setup.</p>
        <p>Click inside any space and it presents ranked, previously-built layouts, reconstructing the chosen one in seconds with approved 3D content and best-practice metadata. It went viral in the construction-tech space and became a research paper at UCL's first DC I/O conference.</p>
    </div>

    <div class="project">
        <h3>AHMM Tools <span class="cv-year">2017 – 2021</span></h3>
        <p class="project-kicker">Custom design and workflow tools for architects</p>
        <img src="{{ '/assets/images/projects/ahmm-tools.jpg' | relative_url }}" alt="AHMM Tools: the element aligner palette beside an annotated construction detail">
        <p>Four years of building whatever <a href="https://www.ahmm.co.uk/">AHMM</a>'s project teams needed, distilled into a plugin deployed to every architect across multiple international offices.</p>
        <p>The greatest hits: an element pre-delete tool showing the knock-on consequences before you break something, an embodied carbon calculator aligned to RICS categories, model issue validation and stripping, geolocation setup, aligners and distributors, data validation checks and instant guidance search.</p>
        <p>Beyond the plugin, I built automation bridging Revit to InDesign, Excel, SQL databases and local knowledge stores — code-signed, silently deployed, and reporting telemetry so our Digital Design Group understood how each tool was actually used.</p>
    </div>

    <div class="project">
        <h3>Computational Chandelier <span class="cv-year">2017 – 2018</span></h3>
        <p class="project-kicker">Math-driven artwork for Broadgate, with sculptor Vlad Tenu</p>
        <img src="{{ '/assets/images/projects/chandelier.jpg' | relative_url }}" alt="The finished Broadgate lighting feature: hundreds of suspended ring lamps forming a rippled canopy in an atrium">
        <p>Sculptor <a href="https://www.vladtenu.com/">Vlad Tenu</a> asked me to build a custom tool to explore ideas for a large lighting feature — the centrepiece of a Broadgate fit-out. We devised a math-driven approach: two colliding sine waves with an exponential function controlling sculptural tension, tunable by lamp density, curve steepness, offsets, depth, even colour temperature.</p>
        <p>Paired with Dynamo and Enscape, the workflow let the sculpture be shaped in real time using VR, ensuring the form worked within the tight lobby space.</p>
    </div>

    <div class="project">
        <h3>Facade Explorer <span class="cv-year">2018 – 2019</span></h3>
        <p class="project-kicker">Instant VR concept optioneering tool</p>
        <img src="{{ '/assets/images/projects/facade-explorer.jpg' | relative_url }}" alt="A matrix of fifteen generated facade options in different materials and colourways">
        <p>An opportunistic mash-up of technologies built for early-stage project bids: a rule-driven Revit family generating an L-System facade with over 60 instance parameters, a Dynamo script recreating the optioneering spirit of Autodesk's Fractal, and Enscape delivering bid-quality visuals.</p>
        <p>A happy side effect: it ran in VR through a game engine, so users could walk around a facade while trying new options every few seconds — 'locking' features they liked and exploring an enormous design space in a playful, engaging way.</p>
    </div>

    <h2 id="open-source">Open Source</h2>

    <div class="project">
        <h3><a href="https://dynamopythonprimer.gitbook.io/dynamo-python-primer">Dynamo Python Primer</a> <span class="cv-year">2019</span></h3>
        <p class="project-kicker">Guidance passion project</p>
        <img src="{{ '/assets/images/projects/python-primer.jpg' | relative_url }}" alt="The Dynamo Python Primer on Gitbook, open at its welcome page">
        <p>Many capable Dynamo users struggle to make the jump from node-based usage to Python and API calls — not because Python is hard, but because nobody explained how IronPython works or what .NET even is.</p>
        <p>Over Christmas 2019 I wrote an <a href="https://github.com/OliverEGreen/DynamoPythonPrimer">open-source guide</a> covering all the knowledge I'd found hard to come by.</p>
        <p>It's been visited tens of thousands of times and was eventually merged into the official guidance by Autodesk.</p>
    </div>

    <div class="project">
        <h3><a href="https://github.com/OliverEGreen/Regular">Regular</a> <span class="cv-year">2019 – 2020</span></h3>
        <p class="project-kicker">Intuitive data validation for architects</p>
        <img src="{{ '/assets/images/projects/regular.jpg' | relative_url }}" alt="Regular's rule manager: building a plain-English data format rule from parts like 'Specific Character' and 'Any Digit'">
        <p>Regular expressions are brilliant for data validation and miserable to write (even for many programmers).</p>
        <p>Born at the 2019 AECTech Hackathon, Regular brings their benefits to non-coders: define data format rules in plain English through an intuitive UI, and the plugin silently translates them into regex.</p>
        <p>Rules live inside Revit models via the Extensible Storage API, and the Dynamic Model Updater validates data continuously against them — flagging each element's compliance with your standards as you work.</p>
    </div>

    <p class="projects-fun-link">Looking for the silly stuff? That lives over in <a href="{{ '/fun/' | relative_url }}">Fun</a>.</p>
</article>
