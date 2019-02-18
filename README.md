**MetaGrasp: Data Efficient Grasping by Affordance Interpreter Network** 

**Junhao Cai, Hui Cheng, Zhanpeng Zhang, Jingcheng Su**

Presented at the 2019 International Conference on Robotics and Automation (ICRA).

**Abstract** Data-driven approach for grasping shows significant advance recently. But these approaches usually require much training data. To increase the efficiency of grasping data collection, this paper presents a novel grasp training system including the whole pipeline from data collection to model inference. The system can collect effective grasp sample with a corrective strategy assisted by antipodal grasp rule, and we design an affordance interpreter network to predict pixelwise grasp affordance map. We define graspability, ungraspability and background as grasp affordances. The key advantage of our system is that the pixel-level affordance interpreter network trained with only a small number of grasp samples under antipodal rule can achieve significant performance on totally unseen objects and backgrounds. The training sample is only collected in simulation. Extensive qualitative and quantitative experiments demonstrate the accuracy and robustness of our proposed approach. In the real-world grasp experiments, we achieve a grasp success rate of 93\% on a set of household items and 91\% on a set of adversarial items with only about 6,300 simulated samples. We also achieve 87\% accuracy in clutter scenario. Although the model is trained using only RGB image, when changing the background textures, it also performs well and can achieve even 94\% accuracy on the set of adversarial objects, which outperforms current state-of-the-art methods.

Preprint: coming soon
Code: comming soon

You can use the [editor on GitHub](https://github.com/sysu-robotics-lab/MetaGrasp/edit/gh-pages/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sysu-robotics-lab/MetaGrasp/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
