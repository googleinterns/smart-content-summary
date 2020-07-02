# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Testing for custom_utils.py."""

import unittest
from unittest import TestCase
import custom_utils
    
    
class CustomUtilsTestCase(TestCase):
    def test_convert_to_POS(self):
        result = custom_utils.convert_to_POS(["This is a test string"])
        self.assertTrue(max(result) <= 40 and min(result) >= 3)
    
    
if __name__ == '__main__':
    unittest.main()
