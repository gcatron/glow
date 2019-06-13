/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "llvm/ADT/STLExtras.h"

#include <future>

using namespace glow;

ExecutionEngine::ExecutionEngine(llvm::StringRef backend) {
  setBackend(backend);
}

/// Set the code generator kind to \p backend.
void ExecutionEngine::setBackend(llvm::StringRef backend) {
  setBackend(createBackend(backend));
}

/// Set the code generator to the given \p backend.
void ExecutionEngine::setBackend(Backend *backend, bool ownsBackend) {
  // bool differentKinds = (backend_ == nullptr || backend == nullptr) ||
  //                       backend->getBackendName() !=
  //                       backend_->getBackendName();
  module_.reset(new Module);
  rawModule_ = module_.get();
  if (ownsBackend_) {
    delete backend_;
  }
  backend_ = backend;
  ownsBackend_ = ownsBackend;
  clear();

  // if (differentKinds) {
  // if (device_) {
  //   EXIT_ON_ERR(device_->stop());
  //   device_.reset();
  // }
  if (hostManager_) {
    EXIT_ON_ERR(hostManager_->clearHost());
    hostManager_.reset();
  }

  if (backend) {
    // device_ = std::unique_ptr<runtime::DeviceManager>(
    //     runtime::DeviceManager::createDeviceManager(
    //         backend->getBackendKind()));
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config =
        llvm::make_unique<runtime::DeviceConfig>(backend->getBackendName());
    configs.push_back(std::move(config));
    hostManager_ = llvm::make_unique<runtime::HostManager>(std::move(configs));
    // EXIT_ON_ERR(device_->init());
  }
  // }
}

const Backend *ExecutionEngine::getBackend() const { return backend_; }

ExecutionEngine::~ExecutionEngine() {
  // Call setBackend to make sure that backend_ is deleted if it's owned.
  setBackend(nullptr, /*ownsBackend*/ false);
}

void ExecutionEngine::clear() {
  // for (auto &func : compiledFunctions_) {
  //   device_->evictNetwork(func.first(), [](std::string, llvm::Error err) {
  //     EXIT_ON_ERR(std::move(err));
  //   });
  // }
  if (hostManager_) {
    EXIT_ON_ERR(hostManager_->clearHost());
  }
  compiledFunctions_.clear();
}

void glow::updateInputPlaceholders(PlaceholderBindings &bindings,
                                   llvm::ArrayRef<Placeholder *> ph,
                                   llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    assert(ph[i] && "Invalid value");
    auto *backingTensor = bindings.get(ph[i]);
    assert(backingTensor && "Can't find the placeholder");
    auto dim = inputs[i]->dims();
    (void)dim;
    assert(backingTensor->getType().isEqual(inputs[i]->getType()) &&
           "Mismatch on Placeholder and Tensor types.");
    backingTensor->assign(inputs[i]);
  }
}

void glow::updateInputPlaceholdersByName(PlaceholderBindings &bindings,
                                         Module *mod,
                                         llvm::ArrayRef<llvm::StringRef> ph,
                                         llvm::ArrayRef<Tensor *> inputs) {
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of Placeholders");

  for (int i = 0, e = ph.size(); i < e; i++) {
    Placeholder *p = mod->getPlaceholderByName(legalizeName(ph[i]));
    Tensor *t = inputs[i];
    assert(t && "Invalid tensor.");
    assert(p && "Invalid placeholder.");
    updateInputPlaceholders(bindings, {p}, {t});
  }
}

void ExecutionEngine::runInternal(ExecutionContext &context,
                                  llvm::StringRef name) {
  // Make sure that the bindings have backing tensors for all placeholders.
  context.getPlaceholderBindings()->allocate(rawModule_->getPlaceholders());

  std::unique_ptr<ExecutionContext> contextPtr(&context);
  std::promise<void> runPromise;
  auto fut = runPromise.get_future();
  llvm::Error runErr = llvm::Error::success();
  hostManager_->runNetwork(
      name, std::move(contextPtr),
      [&runPromise, &runErr](runtime::RunIdentifierTy, llvm::Error err,
                             std::unique_ptr<ExecutionContext> contextPtr) {
        // Don't delete context.
        contextPtr.release();
        runErr = std::move(err);
        runPromise.set_value();
      });
  // EXIT_ON_ERR(hostManager_->runNetworkBlocking(
  //     name, *context.getPlaceholderBindings()));

  fut.wait();
  EXIT_ON_ERR(std::move(runErr));
}

void ExecutionEngine::run(ExecutionContext &context) {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  runInternal(context, *compiledFunctions_.begin());
}

void ExecutionEngine::run(ExecutionContext &context, llvm::StringRef name) {
  runInternal(context, name);
}

void ExecutionEngine::run(PlaceholderBindings &bindings) {
  assert(compiledFunctions_.size() == 1 &&
         "Expected exactly one compiled function.");
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, *compiledFunctions_.begin());
  // don't delete bindings
  context.movePlaceholderBindings().release();
}

void ExecutionEngine::run(PlaceholderBindings &bindings, llvm::StringRef name) {
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  runInternal(context, name);
  // don't delete bindings
  context.movePlaceholderBindings().release();
}

void glow::runBatch(ExecutionEngine &EE, PlaceholderBindings &bindings,
                    size_t iterations, size_t &sampleCounter,
                    llvm::ArrayRef<Placeholder *> ph,
                    llvm::ArrayRef<Tensor *> inputs, llvm::StringRef name) {
  // This is the size of one batch (the number of samples in the batch).
  size_t batchSize = ph[0]->getType()->dims()[0];

  assert(!inputs.empty() && "No inputs");
  assert(inputs.size() == ph.size() &&
         "The number of inputs does not match the number of placeholders");

  // For each iteration in the batch:
  for (size_t j = 0; j < iterations; j++) {

    // Update the input placeholders.
    for (int i = 0, e = ph.size(); i < e; i++) {
      assert(ph[i] && "Invalid value");
      auto *backingTensor = bindings.get(ph[i]);
      assert(backingTensor && "Can't find the backing tensor");

      auto dim = inputs[i]->dims();
      assert(backingTensor->dims().drop_front() == dim.drop_front() &&
             "Invalid slice size");
      // Extract the n'th slice, that must be a tensor.
      size_t slc = sampleCounter % dim[0];
      // Pick up one slice from the input tensors, and load it into the
      // corresponding network Placeholder.
      backingTensor->copyConsecutiveSlices(inputs[i], slc);
    }

    // Run the network.
    if (name == "") {
      EE.run(bindings);
    } else {
      EE.run(bindings, name);
    }
    sampleCounter += batchSize;
  }
}

void ExecutionEngine::compile(CompilationMode mode, Function *F,
                              bool clearOtherFunctions) {
  CompilationContext cctx;
  cctx.compMode = mode;
  compile(F, cctx, clearOtherFunctions);
}

void ExecutionEngine::compile(Function *F, CompilationContext &cctx,
                              bool clearOtherFunctions) {
  llvm::StringRef name = F->getName();
  (void)name;
  if (clearOtherFunctions) {
    compiledFunctions_.clear();
  }
  assert(!compiledFunctions_.count(name) &&
         "A function with this name has already been compiled.");
  assert(module_.get() && "Compile has already been called.");

  for (auto &function : module_->getFunctions()) {
    compiledFunctions_.insert(function->getName());
  }

  EXIT_ON_ERR(hostManager_->addNetwork(std::move(module_), cctx));
}
void ExecutionEngine::compile(CompilationMode mode) {
  for (auto &function : module_->getFunctions()) {
    assert(!compiledFunctions_.count(function->getName()) &&
           "A function with this name has already been compiled.");
    compiledFunctions_.insert(function->getName());
  }
  assert(module_.get() && "Compile has already been called.");
  CompilationContext cctx;
  cctx.compMode = mode;
  EXIT_ON_ERR(hostManager_->addNetwork(std::move(module_), cctx));
}
