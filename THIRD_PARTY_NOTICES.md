# Third-party notices

This repository contains original code together with an implementation of
network components that follow published third-party work, and it
redistributes a small subset of a third-party dataset.  This file records the
attribution and the applicable licenses.

The original code, trained weights, and documentation are released under the
Research Use Only License in [`LICENSE`](LICENSE) (non-commercial research and
educational use).  Nothing in that license overrides the upstream terms
below; third-party components remain under their own licenses, and the
third-party data remain under their own terms.

## 1. RAFT-PIV

**Work.** C. Lagemann, K. Lagemann, S. Mukherjee, W. Schröder,
*Deep recurrent optical flow learning for particle image velocimetry data*,
Nature Machine Intelligence **3**, 641–651 (2021),
doi:[10.1038/s42256-021-00369-0](https://doi.org/10.1038/s42256-021-00369-0).

**Code.** The authors' implementation is distributed as a Code Ocean compute
capsule: <https://codeocean.com/capsule/7226151/tree/v1>.

**What is used here.** The overall iterative PIV architecture, the all-pairs
correlation pyramid with four levels and radius four, the motion-feature
encoder, the separable ConvGRU update block, the flow head, the 9x9 convex
upsampling mask, and the 1/4-resolution flow representation follow RAFT-PIV.
These components are isolated in [`raft_piv.py`](raft_piv.py); `model.py` only
assembles them with the original block from `convnext_original.py`.  No source
file from the capsule is redistributed in this repository; the implementation
was written for this release against the published architecture and the
released checkpoint's parameter layout.

**License.** The Code Ocean capsule consulted for this work did not carry an
explicit open-source license file.  RAFT-PIV is derived from RAFT (below),
which is released under the BSD 3-Clause license; the BSD 3-Clause notice is
reproduced in section 2 and applies to the RAFT-derived components.  Users who
intend to redistribute RAFT-PIV source code itself should obtain terms from
the RAFT-PIV authors.

**Data.** The benchmark images used in this release come from the Cai et al.
PIV dataset, not from the RAFT-PIV capsule; see section 4 for its terms.

## 2. RAFT

**Work.** Z. Teed, J. Deng, *RAFT: Recurrent All-Pairs Field Transforms for
Optical Flow*, ECCV 2020, doi:[10.1007/978-3-030-58536-5_24](https://doi.org/10.1007/978-3-030-58536-5_24).

**Code.** <https://github.com/princeton-vl/RAFT>

**License.** BSD 3-Clause License.

```
BSD 3-Clause License

Copyright (c) 2020, princeton-vl
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## 3. ConvNeXt

**Work.** Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie,
*A ConvNet for the 2020s*, CVPR 2022,
doi:[10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167).

**Code.** <https://github.com/facebookresearch/ConvNeXt>

**What is used here.** The design of the depth-wise 7x7 convolution,
channel-last LayerNorm, inverted bottleneck and LayerScale in
`ConvNeXtBlock` was guided by ConvNeXt.  The block used in this work is the
authors' own implementation (see the statement of original code in
[`README.md`](README.md)); no ConvNeXt source file is redistributed.

**License.** MIT License.

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

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

## 4. PIV benchmark data

The files under `examples/five_cases/` are a minimal excerpt from the public
PIV benchmark dataset accompanying:

* S. Cai, S. Zhou, C. Xu, Q. Gao, *Dense motion estimation of particle images
  via a convolutional neural network*, Experiments in Fluids **60**, 73
  (2019), doi:[10.1007/s00348-019-2717-2](https://doi.org/10.1007/s00348-019-2717-2).
* Dataset repository: <https://github.com/shengzesnail/PIV_dataset>.

According to the dataset's README, the data are **provided for research
purposes only; all rights reserved; any commercial use requires the authors'
consent**.  They are **not** covered by the repository license.
For the full dataset, use the official repository and follow its terms.

The 1997 vortex-pair and PIV Challenge jet-flow data used in the paper are not
redistributed in this repository.  Their public sources are listed in
[`README.md`](README.md#public-datasets-used-in-the-paper).

## 5. Software dependencies

These packages are imported at run time but are not redistributed here.  Their
licenses apply to the installed packages:

| package | license | project |
| --- | --- | --- |
| PyTorch | BSD-style | <https://github.com/pytorch/pytorch> |
| NumPy | BSD 3-Clause | <https://github.com/numpy/numpy> |
| tifffile | BSD 3-Clause | <https://github.com/cgohlke/tifffile> |
| Pillow | HPND | <https://github.com/python-pillow/Pillow> |
