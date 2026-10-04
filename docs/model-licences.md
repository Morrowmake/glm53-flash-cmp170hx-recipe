# Model licences

Model weights are separate from the MIT recipe and Apache-2.0 engine. Neither
software licence overrides a checkpoint's terms. Downloads come directly from
Hugging Face; the recipe does not distribute weights in its container image.
The downloader follows each repository's current default revision unless files
are already present. An engine/image pin does not pin those downloaded weights.
Retain the downloaded cards and record the actual checkpoint revision if you
redistribute a permitted copy.

## GLM-5.3-Flash W4A16 target

[canada-quant](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP)
provides the quantisation and declares MIT inherited from
[Z.ai's base model](https://huggingface.co/zai-org/GLM-5.3-Flash).
The quant repository's card links those terms but does not contain their full
copyright/grant/disclaimer. The complete inherited notice is preserved here
from the [base-model licence](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/LICENSE).
Include it and the quantiser's supplied notices with any redistributed copy.

```text
MIT License

Copyright (c) 2026 Z.AI Co., Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DFlash2 drafter

[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/README.md)
is CC BY-NC-ND 4.0, not MIT or Apache-2.0. Its card describes research and
evaluation use and directs commercial licensing requests to contact@inco.ai.
Commercial use requires separate permission from the licensor. The grant does
not permit sharing adapted checkpoints. It permits technical modifications
necessary to exercise licensed rights; not every runtime transformation is an
adaptation. See the [full legal terms](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en),
especially sections 2 and 3.

If sharing an allowed original copy, retain the supplied creator, copyright,
licence, disclaimer and source information, and indicate any permitted
modifications. Keep the [model card](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/bf582e4eacc1810f76656d1811693ff6c6737d2a/README.md)
and the [licence link](https://creativecommons.org/licenses/by-nc-nd/4.0/)
with that copy. The checkpoint's terms do not automatically determine the
licence of generated text.

Method credit: Inco AI, [DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/)
(2026), and Jian Chen, Yesheng Liang and Zhijian Liu,
[DFlash: Block Diffusion for Flash Speculative Decoding](https://github.com/z-lab/dflash)
(ICML 2026), as requested by the checkpoint card.
