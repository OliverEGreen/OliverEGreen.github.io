---
title: Dynamo UI Rebuild
year: 2021
kicker: Contract UI rebuild of visual programming software, for Autodesk
section: work
order: 3
image: /assets/images/projects/dynamo-rebuild.jpg
link: https://dynamobim.org/dynamo-core-2-13-release-part-1-3/
---

<img src="{{ '/assets/images/projects/dynamo-rebuild.jpg' | relative_url }}" alt="The rebuilt Dynamo node interface showing code blocks, node connections and context menus">

<p>A full-circle moment: I first learned to program in Dynamo's interface, and much of this contract's initial workload came from feature requests I had logged on GitHub years earlier.</p>
<p>Working with Autodesk's product and UI team, I rebuilt the core app UI — the highly configurable node UI, the node library and search, much of the package manager, and a new hierarchical warning system — in XAML, C# and WPF's MVVM architecture, with every build subjected to over 10,000 tests. The <a href="https://dynamobim.org/dynamo-core-2-13-release-part-1-3/">Dynamo 2.13 release</a> shipped the results.</p>
<img src="{{ '/assets/images/projects/dynamo-canvas.jpg' | relative_url }}" alt="The rebuilt Dynamo home canvas: library panel, geometry grid and run controls">
<p>The richer UI initially brought a performance hit ("Dynamo is pushing the reasonable limits of what WPF can achieve"), so I profiled it with dotMemory until the rebuilt app outperformed the old one. Autodesk offered me a job afterwards — but my heart was set on product management.</p>
