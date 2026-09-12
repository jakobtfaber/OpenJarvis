import { describe, expect, it, vi } from 'vitest';

vi.mock('../lib/store', () => ({ useAppStore: vi.fn() }));

import { localAccountPickerState } from './DataSourcesPage';
import type { LocalAccountOption } from '../types/connectors';

const account: LocalAccountOption = {
  account_id: 'ACCOUNT-ID',
  protocol: 'imap',
  mail_version: 'V10',
};

describe('account-scoped local connector picker', () => {
  it('stays not ready while the accounts are still being discovered', () => {
    const state = localAccountPickerState('Apple Mail', null, '');

    expect(state.ready).toBe(false);
    expect(state.hint).toContain('Looking for Apple Mail accounts');
  });

  it('explains the Full Disk Access fix when no account is found', () => {
    const state = localAccountPickerState('Apple Mail', [], '');

    expect(state.ready).toBe(false);
    expect(state.hint).toContain('Full Disk Access');
  });

  it('reports an unreadable store instead of the Full Disk Access advice', () => {
    // Discovery failing is not the same as the Mac holding no accounts, and
    // the two need different fixes.
    const state = localAccountPickerState(
      'Apple Mail',
      [],
      '',
      'Could not read the Apple Mail index at /Users/x/Library/Mail/V10',
    );

    expect(state.ready).toBe(false);
    expect(state.hint).toContain('Could not read');
    expect(state.hint).not.toContain('Full Disk Access');
  });

  it('is ready only once an account is picked', () => {
    // The backend rejects a connect request with no account id, so an empty
    // selection must never reach it.
    expect(localAccountPickerState('Apple Mail', [account], '').ready).toBe(false);
    expect(localAccountPickerState('Apple Mail', [account], 'ACCOUNT-ID').ready).toBe(true);
  });

  it('says the other accounts are never read', () => {
    const state = localAccountPickerState('Apple Mail', [account], 'ACCOUNT-ID');

    expect(state.hint).toContain('read-only');
    expect(state.hint).toContain('never read');
  });
});
